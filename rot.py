import os
import torch
import torch.nn.functional as F
import argparse
import json
import glob
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)

layer_cache = {}

ALPHA = 0.0
BETA = 0.0
GAMMA = 0.0
SMOOTH_START_LAYER = 0

last_layer_h = {}

TAU = 0.0

def load_chair_dataset(img_dir, num_samples):
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No jpg images found in {img_dir}.")
    img_paths = sorted(img_paths)
    dataset = []
    for path in img_paths[:num_samples]:
        img = Image.open(path).convert('RGB')
        img_id = os.path.basename(path).split('.')[0]
        dataset.append({
            'image': img,
            'image_id': img_id,
            'question': "Please describe this image in detail.",
            'label': 'none'
        })
    return dataset

def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lang_model = model.model.language_model
        if hasattr(lang_model, 'layers'): return lang_model.layers
        if hasattr(lang_model, 'model') and hasattr(lang_model.model, 'layers'): return lang_model.model.layers
    candidates = ['language_model.model.layers', 'language_model.layers', 'model.layers', 'layers', 'base_model.layers']
    for cand in candidates:
        parts = cand.split('.')
        obj = model
        try:
            for p in parts: obj = getattr(obj, p)
            if hasattr(obj, '__len__') and len(obj) > 0: return obj
        except AttributeError: continue
    raise AttributeError("Unable to locate the layers attribute of the model")

def get_layer_input_hook(module, inputs, layer_idx):
    seq_len = inputs[0].shape[1]
    if seq_len > 1:
        layer_cache[f"layer_input_{layer_idx}"] = inputs[0].detach()
    else:
        if f"layer_input_{layer_idx}" in layer_cache:
            layer_cache[f"layer_input_{layer_idx}"] = torch.cat(
                [layer_cache[f"layer_input_{layer_idx}"], inputs[0].detach()], dim=1
            )

def get_attn_weights_hook(module, inputs, outputs, layer_idx):
    if len(outputs) > 1 and outputs[1] is not None:
        layer_cache[f"attn_weights_{layer_idx}"] = outputs[1].detach()

def modify_mlp_input_hook(module, inputs, layer_idx, v_start, v_end, input_len):
    h_k_full = inputs[0]

    layer_input = layer_cache.get(f"layer_input_{layer_idx}")
    attn_weights = layer_cache.get(f"attn_weights_{layer_idx}")

    if layer_input is None or attn_weights is None:
        return inputs

    current_attn = attn_weights[:, :, -1, :].mean(dim=1)
    total_seq_len = current_attn.shape[1]

    if total_seq_len <= input_len or v_start >= v_end:
        return inputs

    visual_mask = torch.zeros(total_seq_len, dtype=torch.bool, device=h_k_full.device)
    visual_mask[v_start:v_end] = True

    text_mask = torch.ones(total_seq_len, dtype=torch.bool, device=h_k_full.device)
    text_mask[v_start:v_end] = False
    text_mask[-1] = False

    attn_v = current_attn[:, visual_mask]
    attn_t = current_attn[:, text_mask]

    layer_input_v = layer_input[:, visual_mask, :]
    layer_input_t = layer_input[:, text_mask, :]

    mass_v = attn_v.sum(dim=-1).clamp(min=1e-9)
    mass_t = attn_t.sum(dim=-1).clamp(min=1e-9)

    attn_v_norm = attn_v / mass_v.unsqueeze(-1)
    attn_t_norm = attn_t / mass_t.unsqueeze(-1)

    C_V = torch.bmm(attn_v_norm.unsqueeze(1), layer_input_v).squeeze(1)
    C_T = torch.bmm(attn_t_norm.unsqueeze(1), layer_input_t).squeeze(1)

    h_k_b = h_k_full[:, -1, :]

    h_k_32 = h_k_b.to(torch.float32)
    C_T_32 = C_T.to(torch.float32)
    C_V_32 = C_V.to(torch.float32)

    S_T = F.cosine_similarity(h_k_32, C_T_32, dim=-1)
    S_V = F.cosine_similarity(h_k_32, C_V_32, dim=-1)

    E = 2.0 - S_T - S_V

    for b in range(h_k_full.shape[0]):
        orig_norm = torch.norm(h_k_32[b], p=2).clamp(min=1e-9)
        u = h_k_32[b] / orig_norm

        if layer_idx < SMOOTH_START_LAYER and E[b] > TAU:
            w_T = C_T_32[b] / torch.norm(C_T_32[b], p=2).clamp(min=1e-9)
            w_V = C_V_32[b] / torch.norm(C_V_32[b], p=2).clamp(min=1e-9)

            dot_TV = torch.dot(w_T, w_V).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta_TV = torch.acos(dot_TV)

            if theta_TV > 1e-4:
                sin_theta_TV = torch.sin(theta_TV)
                theta_alpha = theta_TV * ALPHA
                s0_TV = torch.sin(theta_TV - theta_alpha) / sin_theta_TV
                s1_TV = torch.sin(theta_alpha) / sin_theta_TV
                C_target = s0_TV * w_T + s1_TV * w_V
            else:
                C_target = w_T
            C_target = C_target / torch.norm(C_target, p=2).clamp(min=1e-9)

            dot_h_target = torch.dot(u, C_target).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta_h = torch.acos(dot_h_target)

            if theta_h > 1e-4:
                sin_theta_h = torch.sin(theta_h)
                theta_beta = theta_h * BETA
                s0_h = torch.sin(theta_h - theta_beta) / sin_theta_h
                s1_h = torch.sin(theta_beta) / sin_theta_h
                u_new = s0_h * u + s1_h * C_target
            else:
                u_new = u
            u_new = u_new / torch.norm(u_new, p=2).clamp(min=1e-9)

            h_new = u_new * orig_norm
            inputs[0][b, -1, :] = h_new.to(inputs[0].dtype)

        elif layer_idx >= SMOOTH_START_LAYER and b in last_layer_h:
            h_prev = last_layer_h[b]
            u_prev = h_prev / torch.norm(h_prev, p=2).clamp(min=1e-9)

            dot_h_prev = torch.dot(u, u_prev).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta_smooth = torch.acos(dot_h_prev)

            if theta_smooth > 1e-4:
                sin_theta_smooth = torch.sin(theta_smooth)
                theta_gamma = theta_smooth * GAMMA
                s0_smooth = torch.sin(theta_smooth - theta_gamma) / sin_theta_smooth
                s1_smooth = torch.sin(theta_gamma) / sin_theta_smooth
                u_new = s0_smooth * u + s1_smooth * u_prev
            else:
                u_new = u

            u_new = u_new / torch.norm(u_new, p=2).clamp(min=1e-9)
            h_new = u_new * orig_norm
            inputs[0][b, -1, :] = h_new.to(inputs[0].dtype)

        last_layer_h[b] = inputs[0][b, -1, :].detach().clone().to(torch.float32)

    return inputs

def cleanup_hook(module, inputs, outputs, layer_idx):
    if f"attn_weights_{layer_idx}" in layer_cache:
        del layer_cache[f"attn_weights_{layer_idx}"]

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Loading processor and model: {args.model_path}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
        processor.tokenizer.padding_side = 'left'

    if args.model_type == "qwen":
        if "2.5" in args.model_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
            )
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    elif args.model_type == "llava":
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="eager"
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="eager"
            )
        image_token_id = model.config.image_token_index if hasattr(model.config, 'image_token_index') else 32000
    else:
        raise ValueError("Unsupported model_type")

    model.eval()
    layers = get_model_layers(model)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset.lower() == "pope":
        dataset_raw = load_dataset("lmms-lab/POPE", name="default", split="test")
        dataset_raw = dataset_raw.filter(lambda x: x['category'] == args.pope_type)
        actual_samples = len(dataset_raw) if args.num_samples == -1 else min(args.num_samples, len(dataset_raw))
        dataset_raw = dataset_raw.select(range(actual_samples))

        dataset = []
        for item in dataset_raw:
            question_text = item.get('text', item.get('question', ''))
            dataset.append({
                'image': item['image'],
                'image_id': str(item.get('image_source', item.get('question_id', f"pope_{len(dataset)}"))),
                'question': question_text,
                'label': item.get('answer', item.get('label', 'none'))
            })
    elif args.dataset.lower() == "chair":
        if not args.coco_img_dir:
            raise ValueError("Running CHAIR evaluation requires specifying the local COCO image path.")
        dataset = load_chair_dataset(args.coco_img_dir, args.num_samples)
    else:
        raise NotImplementedError("pope or chair")

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    out_file = open(args.output_file, "w", encoding='utf-8')

    batch_size = args.batch_size
    hooks = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Intervention inference in progress"):
        batch_data = dataset[i : i + batch_size]
        images = [item['image'] for item in batch_data]
        questions = [item['question'] for item in batch_data]

        if args.dataset.lower() == "pope":
            cot_suffix = "Please describe the image content concisely, and then answer with 'yes' or 'no' at the end."
            processed_questions = [q + cot_suffix for q in questions]
        else:
            processed_questions = questions

        if args.model_type == "qwen":
            texts = []
            for q in processed_questions:
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        elif args.model_type == "llava":
            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in processed_questions]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[1]
        v_starts, v_ends = [], []
        for b in range(len(batch_data)):
            indices = (inputs["input_ids"][b] == image_token_id).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                v_starts.append(indices[0].item())
                v_ends.append(indices[-1].item() + 1)
            else:
                v_starts.append(0)
                v_ends.append(0)

        for h in hooks: h.remove()
        hooks.clear()
        layer_cache.clear()
        last_layer_h.clear()

        for l_idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_pre_hook(lambda m, inp, idx=l_idx: get_layer_input_hook(m, inp, idx)))
            hooks.append(layer.self_attn.register_forward_hook(lambda m, inp, out, idx=l_idx: get_attn_weights_hook(m, inp, out, idx)))
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, inp, idx=l_idx: modify_mlp_input_hook(m, inp, idx, v_starts[0], v_ends[0], input_len)
            ))
            hooks.append(layer.register_forward_hook(lambda m, inp, out, idx=l_idx: cleanup_hook(m, inp, out, idx)))

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False
            )

        output_sequences = output_ids.sequences
        input_lens = [len(inputs["input_ids"][j]) for j in range(len(batch_data))]
        for j in range(len(batch_data)):
            generated_ids = output_sequences[j][input_lens[j]:]
            response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            final_text = response
            if args.dataset.lower() == "pope":
                import re
                matches = re.findall(r'\b(yes|no)\b', response.lower())
                final_text = matches[-1] if matches else "no"

            result_dict = {
                "image_id": batch_data[j]['image_id'],
                "question": processed_questions[j],
                "label": batch_data[j]['label'],
                "text": final_text,
                "full_text": response
            }
            out_file.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            out_file.flush()

    out_file.close()
    for h in hooks: h.remove()
    print(f"Inference completed, results with intervention saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--model_type", type=str, required=True, choices=["llava", "qwen"], help="Model type")
    parser.add_argument("--dataset", type=str, default="pope", choices=["pope", "chair"])
    parser.add_argument("--pope_type", type=str, default="adversarial", choices=["random", "popular", "adversarial"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output_file", type=str, default="rot_output.jsonl")
    parser.add_argument("--coco_img_dir", type=str, default="", help="COCO validation set local image path")
    main(parser.parse_args())

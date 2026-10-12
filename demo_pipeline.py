"""
PURE (Princeton University Relation Extraction) End-to-End Pipeline Demo
Paper: A Frustratingly Easy Approach for Entity and Relation Extraction (NAACL 2021)
Authors: Zexuan Zhong, Danqi Chen

This standalone script runs inference on CPU without legacy allennlp dependencies.
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel

# -----------------------------------------------------------------------------
# 1. Standalone Model Architectures (Pure PyTorch, 0 external legacy dependency)
# -----------------------------------------------------------------------------

def batched_index_select(target, indices):
    """
    target:  [batch_size, seq_len, hidden_size]
    indices: [batch_size, num_indices]
    returns: [batch_size, num_indices, hidden_size]
    """
    dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), target.size(2))
    return torch.gather(target, 1, dummy)

class FeedForward(nn.Module):
    """Matches the exact state_dict structure of allennlp.modules.FeedForward"""
    def __init__(self, input_dim, hidden_dims=150, dropout=0.2):
        super().__init__()
        self._linear_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dims),
            nn.Linear(hidden_dims, hidden_dims)
        ])
        self._dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self._linear_layers:
            x = F.relu(layer(x))
            x = self._dropout(x)
        return x

class BertForEntity(nn.Module):
    def __init__(self, config, num_ner_labels=7, head_hidden_dim=150, width_embedding_dim=150, max_span_length=8):
        super().__init__()
        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim, hidden_dims=head_hidden_dim, dropout=0.2),
            nn.Linear(head_hidden_dim, num_ner_labels)
        )

    def forward(self, input_ids, spans, attention_mask=None):
        """
        spans: [batch_size, num_spans, 3] -> (start_idx, end_idx, span_len)
        """
        seq_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        seq_output = self.hidden_dropout(seq_output)

        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_width = spans[:, :, 2].view(spans.size(0), -1)

        # Mathematical Span Representation: g(s_i) = [x_start ; x_end ; phi(span_len)]
        spans_start_emb = batched_index_select(seq_output, spans_start)
        spans_end_emb = batched_index_select(seq_output, spans_end)
        spans_width_emb = self.width_embedding(spans_width)

        spans_emb = torch.cat((spans_start_emb, spans_end_emb, spans_width_emb), dim=-1)
        logits = self.ner_classifier(spans_emb)
        return logits

class BertForRelation(nn.Module):
    def __init__(self, config, num_rel_labels=8):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, num_rel_labels)

    def forward(self, input_ids, attention_mask=None, sub_idx=None, obj_idx=None):
        seq_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        # Attention Sink Extraction: h(s_i, s_j) = [x_<S:Type> ; x_<O:Type>]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(seq_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(seq_output, obj_idx)])
        
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        return logits


# -----------------------------------------------------------------------------
# 2. Pipeline Manager
# -----------------------------------------------------------------------------

class PUREPipeline:
    def __init__(self, models_dir):
        self.ent_model_dir = os.path.join(models_dir, "ent-scib-ctx0")
        self.rel_model_dir = os.path.join(models_dir, "rel-scib-ctx0")

        # Labels
        self.ner_id2label = {
            0: "None", 1: "Method", 2: "OtherScientificTerm",
            3: "Task", 4: "Generic", 5: "Material", 6: "Metric"
        }
        self.rel_id2label = {
            0: "no_relation", 1: "PART-OF", 2: "USED-FOR", 3: "FEATURE-OF",
            4: "CONJUNCTION", 5: "EVALUATE-FOR", 6: "HYPONYM-OF", 7: "COMPARE"
        }

        print("Loading SciBERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.ent_model_dir)

        # Load Typed Marker special tokens
        special_tokens_path = os.path.join(self.rel_model_dir, "special_tokens.json")
        with open(special_tokens_path, 'r', encoding='utf-8') as f:
            self.special_tokens = json.load(f)

        print("Loading Entity Model (ent-scib-ctx0)...")
        ent_cfg = BertConfig.from_json_file(os.path.join(self.ent_model_dir, "config.json"))
        self.ent_model = BertForEntity(ent_cfg, num_ner_labels=len(self.ner_id2label))
        ent_sd = torch.load(os.path.join(self.ent_model_dir, "pytorch_model.bin"), map_location="cpu")
        self.ent_model.load_state_dict(ent_sd, strict=True)
        self.ent_model.eval()

        print("Loading Relation Model (rel-scib-ctx0)...")
        rel_cfg = BertConfig.from_json_file(os.path.join(self.rel_model_dir, "config.json"))
        self.rel_model = BertForRelation(rel_cfg, num_rel_labels=len(self.rel_id2label))
        rel_sd = torch.load(os.path.join(self.rel_model_dir, "pytorch_model.bin"), map_location="cpu")
        self.rel_model.load_state_dict(rel_sd, strict=True)
        self.rel_model.eval()
        print("Models successfully loaded on CPU!\n" + "="*70)

    def run(self, raw_tokens, max_span_length=8):
        print("\n" + "="*70)
        print("【原始输入句子 (Raw Input Sentence)】")
        print("  Tokens :", " ".join(raw_tokens))
        print("  长度   :", len(raw_tokens), "tokens")
        print("="*70)

        # --- Stage 1: Entity Recognition ---
        print("\n" + "-"*70)
        print("▶ 第一阶段：实体跨度模型 (Span-based Entity Model)")
        print("-"*70)

        # Tokenization & Alignment
        # 保存每个词token化后的内容（一个词可能被分出几个token）
        subwords = ["[CLS]"]
        # 记录每个词token化后在subwords中的起止位置
        tok_to_subword = []
        for tok in raw_tokens:
            sw = self.tokenizer.tokenize(tok)
            if not sw:
                sw = [self.tokenizer.unk_token]
            tok_to_subword.append((len(subwords), len(subwords) + len(sw) - 1))
            subwords.extend(sw)
        subwords.append("[SEP]")

        # 将token转换为tensor向量
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(subwords)])
        attn_mask = torch.ones_like(input_ids)

        # Candidate span enumeration (length <= max_span_length)
        candidate_spans = []
        n_tokens = len(raw_tokens)
        for length in range(1, max_span_length + 1):
            for i in range(n_tokens - length + 1):
                j = i + length - 1
                sw_start = tok_to_subword[i][0]
                sw_end = tok_to_subword[j][1]
                candidate_spans.append([sw_start, sw_end, length, i, j])

        spans_tensor = torch.tensor([[s[:3] for s in candidate_spans]])

        print(f"  • 枚举跨度总数 (长度 1~{max_span_length}) : {len(candidate_spans)} 个候选跨度")
        print(f"  • 跨度微观表征公式 : g(s_i) = [x_START ; x_END ; phi(length)]")

        with torch.no_grad():
            ner_logits = self.ent_model(input_ids, spans_tensor, attn_mask)
            ner_probs = F.softmax(ner_logits[0], dim=-1)
            pred_classes = torch.argmax(ner_probs, dim=-1)

        detected_entities = []
        for idx, cls_id in enumerate(pred_classes.tolist()):
            if cls_id != 0: # Not 'None'
                label = self.ner_id2label[cls_id]
                prob = ner_probs[idx, cls_id].item()
                raw_i, raw_j = candidate_spans[idx][3], candidate_spans[idx][4]
                entity_text = " ".join(raw_tokens[raw_i : raw_j + 1])
                detected_entities.append({
                    "start": raw_i, "end": raw_j,
                    "text": entity_text, "label": label, "prob": prob
                })

        print("\n  【实体预测结果 (Predicted Entities)】")
        for ent in detected_entities:
            print(f"    ✓ [{ent['start']}, {ent['end']}] \"{ent['text']}\" -> 【{ent['label']}】 (置信度: {ent['prob']*100:.1f}%)")

        if len(detected_entities) < 2:
            print("  (候选实体少于 2 个，无需进行关系配对)")
            return

        # --- Stage 2: Relation Extraction with Typed Entity Markers ---
        print("\n" + "-"*70)
        print("▶ 第二阶段：带类型实体标记关系模型 (Typed Entity Marker)")
        print("-"*70)

        # Generate candidate pairs
        pairs = []
        for i, sub in enumerate(detected_entities):
            for j, obj in enumerate(detected_entities):
                if i != j:
                    pairs.append((sub, obj))

        print(f"  • 候选实体对总数 : {len(pairs)} 对 (O(k^2) 量级)")

        predicted_relations = []
        for p_idx, (sub, obj) in enumerate(pairs):
            # Insert Typed Markers
            sub_start_marker = self.special_tokens.get(f"SUBJ_START={sub['label']}", "[unused1]")
            sub_end_marker = self.special_tokens.get(f"SUBJ_END={sub['label']}", "[unused2]")
            obj_start_marker = self.special_tokens.get(f"OBJ_START={obj['label']}", "[unused3]")
            obj_end_marker = self.special_tokens.get(f"OBJ_END={obj['label']}", "[unused4]")

            # Reconstruct sequence with markers
            marked_tokens = ["[CLS]"]
            sub_sw_idx = -1
            obj_sw_idx = -1

            for tok_idx, tok in enumerate(raw_tokens):
                if tok_idx == sub["start"]:
                    sub_sw_idx = len(marked_tokens)
                    marked_tokens.append(sub_start_marker)
                if tok_idx == obj["start"]:
                    obj_sw_idx = len(marked_tokens)
                    marked_tokens.append(obj_start_marker)

                # add token subwords
                sw = self.tokenizer.tokenize(tok)
                if not sw:
                    sw = [self.tokenizer.unk_token]
                marked_tokens.extend(sw)

                if tok_idx == sub["end"]:
                    marked_tokens.append(sub_end_marker)
                if tok_idx == obj["end"]:
                    marked_tokens.append(obj_end_marker)

            marked_tokens.append("[SEP]")

            rel_input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(marked_tokens)])
            rel_sub_idx = torch.tensor([sub_sw_idx])
            rel_obj_idx = torch.tensor([obj_sw_idx])

            with torch.no_grad():   
                rel_logits = self.rel_model(rel_input_ids, sub_idx=rel_sub_idx, obj_idx=rel_obj_idx)
                rel_probs = F.softmax(rel_logits[0], dim=-1)
                pred_rel_id = torch.argmax(rel_probs).item()

            rel_name = self.rel_id2label[pred_rel_id]
            prob = rel_probs[pred_rel_id].item()

            if rel_name != "no_relation":
                predicted_relations.append({
                    "sub": sub["text"], "sub_label": sub["label"],
                    "obj": obj["text"], "obj_label": obj["label"],
                    "relation": rel_name, "prob": prob
                })

            print(f"\n  [实体对 {p_idx+1}] 主语: \"{sub['text']}\"({sub['label']}) -> 宾语: \"{obj['text']}\"({obj['label']})")
            print(f"    • 注入标记后的输入序列 :")
            preview_str = " ".join(marked_tokens[1:-1])
            preview_clean = preview_str.replace(sub_start_marker, f"<S:{sub['label']}>")\
                                       .replace(sub_end_marker, f"</S:{sub['label']}>")\
                                       .replace(obj_start_marker, f"<O:{obj['label']}>")\
                                       .replace(obj_end_marker, f"</O:{obj['label']}>")
            print(f"      {preview_clean}")
            print(f"    • 关系模型预测结果 : 【{rel_name}】 (置信度: {prob*100:.1f}%)")

        print("\n" + "="*70)
        print("【端到端三元组最终抽取结果 (Final Triplet Results)】")
        if predicted_relations:
            for rel in predicted_relations:
                print(f"  🎯 ({rel['sub']})  ──[{rel['relation']}]──>  ({rel['obj']})  [置信度: {rel['prob']*100:.1f}%]")
        else:
            print("  (未检测到预定义关系)")
        print("="*70)

        # --- Stage 3: Fast Approximation Visualizer ---
        print("\n" + "-"*70)
        print("▶ 论文核心创新透视：非对称定向注意力掩码 (Attention Mask Visualizer)")
        print("-"*70)
        print("论文 Figure 1(c) 批处理机制：如何用一个二维掩码矩阵把 O(k^2) 次前向压缩为 1 次前向？")
        self.visualize_attention_mask(raw_tokens, detected_entities)

    def visualize_attention_mask(self, raw_tokens, entities):
        # We demonstrate with 2 pairs
        pairs = []
        for i, sub in enumerate(entities):
            for j, obj in enumerate(entities):
                if i != j:
                    pairs.append((sub, obj))
        if len(pairs) > 2:
            pairs = pairs[:2]

        tokens_seq = ["CLS"] + raw_tokens[:4] + ["..."]
        text_len = len(tokens_seq)
        
        # Add 4 markers for pair 1, 4 markers for pair 2
        p1_markers = ["<S1>", "</S1>", "<O1>", "</O1>"]
        p2_markers = ["<S2>", "</S2>", "<O2>", "</O2>"]
        all_tokens = tokens_seq + p1_markers + p2_markers
        n = len(all_tokens)

        # Build attention mask matrix:
        # text tokens only attend to text
        # p1 markers attend to text + p1 markers
        # p2 markers attend to text + p2 markers
        matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if j < text_len:
                    matrix[i][j] = 1 # anyone can see text tokens
                elif text_len <= i < text_len + 4 and text_len <= j < text_len + 4:
                    matrix[i][j] = 1 # p1 can see p1
                elif i >= text_len + 4 and j >= text_len + 4:
                    matrix[i][j] = 1 # p2 can see p2

        print("\n  二维定向注意力掩码矩阵 (1 = 可见关注, 0 = 掩蔽屏蔽)：")
        header = " " * 8 + " ".join([f"{t:>5}" for t in all_tokens])
        print(header)
        for i, row in enumerate(matrix):
            row_str = " ".join([f"{v:>5}" for v in row])
            print(f"{all_tokens[i]:>7} : {row_str}")

        print("\n  💡【机理验证结论】：")
        print("    1. 正文 Token（前几行）对所有 <S1>, <O1>, <S2>, <O2> 均为 0（纯净，不受 Marker 污染）！")
        print("    2. 第 1 对 Marker 只能看到正文和自己的队友，对第 2 对 Marker 全为 0（互不串扰）！")
        print("    3. 这就是为什么可以在【单次前向】中并行完成整句所有实体对的关系预测！")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "scierc_models")

    pipeline = PUREPipeline(models_dir)

    # Figure 1 classic sentence from SciERC (NAACL 2021)
    sentence = ["MORPA", "is", "a", "fully", "implemented", "parser", "for", "text-to-speech"]
    pipeline.run(sentence)

if __name__ == "__main__":
    main()

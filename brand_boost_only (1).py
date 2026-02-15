import pandas as pd
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("Loading data...", flush=True)
base_path = "/home/milkak/A-LLMRec/data/amazon"
train = pd.read_csv(f"{base_path}/All_Beauty.train.csv")

meta_items = {}
with open(f"{base_path}/meta_All_Beauty.jsonl") as f:
    for line in f:
        item = json.loads(line)
        asin = item['parent_asin']
        title = item.get('title', '')
        desc = item.get('description', '') if isinstance(item.get('description', ''), str) else ''
        brand = item.get('details', {}).get('Brand', '')
        category = item.get('categories', [])
        category = category[0] if category else ''
        store = item.get('store', '')
        meta_items[asin] = {
            'text': (title + ' ' + desc).strip(),
            'brand': str(brand).lower().strip(),
            'category': str(category).lower().strip(),
            'store': str(store).lower().strip(),
        }

train = train[train['parent_asin'].isin(meta_items.keys())]
train_sorted = train.sort_values(['user_id', 'timestamp'])
user_seqs = train_sorted.groupby('user_id')['parent_asin'].apply(list).to_dict()
item_popularity = train['parent_asin'].value_counts().to_dict()

print("Building statistics...", flush=True)

item_next = defaultdict(lambda: defaultdict(float))
brand_next_items = defaultdict(lambda: defaultdict(float))
category_next_items = defaultdict(lambda: defaultdict(float))
item_cooccur = defaultdict(lambda: defaultdict(float))
item_bought_after = defaultdict(lambda: defaultdict(float))

for seq in user_seqs.values():
    for i in range(1, len(seq)):
        prev_item, cur_item = seq[i-1], seq[i]
        item_next[prev_item][cur_item] += 1
        item_bought_after[prev_item][cur_item] += 1
        
        prev_brand = meta_items.get(prev_item, {}).get('brand', '')
        if prev_brand:
            brand_next_items[prev_brand][cur_item] += 1
        
        prev_cat = meta_items.get(prev_item, {}).get('category', '')
        if prev_cat:
            category_next_items[prev_cat][cur_item] += 1
    
    items = list(set(seq))
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i != j:
                item_cooccur[a][b] += 1

brand_popular = defaultdict(list)
category_popular = defaultdict(list)
store_popular = defaultdict(list)

for asin, info in meta_items.items():
    pop = item_popularity.get(asin, 0)
    if info['brand']:
        brand_popular[info['brand']].append((asin, pop))
    if info['category']:
        category_popular[info['category']].append((asin, pop))
    if info['store']:
        store_popular[info['store']].append((asin, pop))

for k in brand_popular:
    brand_popular[k] = [x[0] for x in sorted(brand_popular[k], key=lambda x: -x[1])]
for k in category_popular:
    category_popular[k] = [x[0] for x in sorted(category_popular[k], key=lambda x: -x[1])]
for k in store_popular:
    store_popular[k] = [x[0] for x in sorted(store_popular[k], key=lambda x: -x[1])]

print("Loading text embeddings...", flush=True)
from sentence_transformers import SentenceTransformer
all_asins = list(meta_items.keys())
texts = [meta_items[a]['text'] for a in all_asins]
st_model = SentenceTransformer('all-mpnet-base-v2')
X_emb = st_model.encode(texts, convert_to_numpy=True, batch_size=256, show_progress_bar=True)
asin2idx = {a: i for i, a in enumerate(all_asins)}
idx2asin = {i: a for a, i in asin2idx.items()}

def get_text_similar(item, history_set, topk=100):
    if item not in asin2idx:
        return {}
    v = X_emb[asin2idx[item]].reshape(1, -1)
    sims = cosine_similarity(v, X_emb).ravel()
    top_idx = np.argsort(sims)[-(topk+50):][::-1]
    result = {}
    for idx in top_idx:
        asin = idx2asin[idx]
        if asin not in history_set and asin != item:
            result[asin] = float(sims[idx])
        if len(result) >= topk:
            break
    return result

def score_single_item(item, history_set):
    scores = defaultdict(float)
    
    brand = meta_items.get(item, {}).get('brand', '')
    category = meta_items.get(item, {}).get('category', '')
    store = meta_items.get(item, {}).get('store', '')
    
    # 1. Direct item transitions - UNCHANGED (10.0)
    for next_item, cnt in item_bought_after[item].items():
        if next_item not in history_set:
            scores[next_item] += cnt * 10.0
    
    # 2. Brand transitions - INCREASED (5.0 -> 6.0)
    if brand:
        for next_item, cnt in brand_next_items[brand].items():
            if next_item not in history_set:
                scores[next_item] += cnt * 6.0
        # Brand popular - INCREASED (0.8 -> 1.0)
        for rank, pop_item in enumerate(brand_popular[brand][:50]):
            if pop_item not in history_set:
                scores[pop_item] += (50 - rank) * 1.0
    
    # 3. Category transitions - UNCHANGED (4.0)
    if category:
        for next_item, cnt in category_next_items[category].items():
            if next_item not in history_set:
                scores[next_item] += cnt * 4.0
        for rank, pop_item in enumerate(category_popular[category][:50]):
            if pop_item not in history_set:
                scores[pop_item] += (50 - rank) * 0.6
    
    # 4. Store - UNCHANGED (0.4)
    if store:
        for rank, pop_item in enumerate(store_popular[store][:30]):
            if pop_item not in history_set:
                scores[pop_item] += (30 - rank) * 0.4
    
    # 5. Text similarity - UNCHANGED (6.0)
    text_sim = get_text_similar(item, history_set, topk=100)
    for sim_item, sim_score in text_sim.items():
        scores[sim_item] += sim_score * 6.0
    
    # 6. Co-occurrence - UNCHANGED (2.0)
    for cooc_item, cnt in item_cooccur[item].items():
        if cooc_item not in history_set:
            scores[cooc_item] += cnt * 2.0
    
    return scores

def score_multi_item(history_items, history_set):
    scores = defaultdict(float)
    last_item = history_items[-1]
    
    brand = meta_items.get(last_item, {}).get('brand', '')
    category = meta_items.get(last_item, {}).get('category', '')
    store = meta_items.get(last_item, {}).get('store', '')
    
    # 1. Sequential - UNCHANGED (8.0)
    for next_item, cnt in item_next[last_item].items():
        if next_item not in history_set:
            scores[next_item] += cnt * 8.0
    
    # 2. Brand - INCREASED (4.0 -> 5.0)
    if brand:
        for next_item, cnt in brand_next_items[brand].items():
            if next_item not in history_set:
                scores[next_item] += cnt * 5.0
        for rank, pop_item in enumerate(brand_popular[brand][:30]):
            if pop_item not in history_set:
                scores[pop_item] += (30 - rank) * 0.6
    
    # 3. Category - UNCHANGED
    if category:
        for next_item, cnt in category_next_items[category].items():
            if next_item not in history_set:
                scores[next_item] += cnt * 3.0
        for rank, pop_item in enumerate(category_popular[category][:30]):
            if pop_item not in history_set:
                scores[pop_item] += (30 - rank) * 0.4
    
    # 4. Store - UNCHANGED
    if store:
        for rank, pop_item in enumerate(store_popular[store][:20]):
            if pop_item not in history_set:
                scores[pop_item] += (20 - rank) * 0.3
    
    # 5. Text similarity - UNCHANGED
    text_sim = get_text_similar(last_item, history_set, topk=50)
    for sim_item, sim_score in text_sim.items():
        scores[sim_item] += sim_score * 3.0
    
    # 6. Co-occurrence - UNCHANGED
    for hist_item in history_items[-5:]:
        for cooc_item, cnt in item_cooccur[hist_item].items():
            if cooc_item not in history_set:
                scores[cooc_item] += cnt * 1.5
    
    # 7. Repeat brand - UNCHANGED
    history_brands = {meta_items.get(it, {}).get('brand', '') for it in history_items}
    for hb in history_brands:
        if hb:
            for rank, pop_item in enumerate(brand_popular[hb][:20]):
                if pop_item not in history_set:
                    scores[pop_item] += (20 - rank) * 0.3
    
    return scores

def ensemble_score(history_items, top_n=10):
    if not history_items:
        return list(item_popularity.keys())[:top_n]
    
    history_set = set(history_items)
    
    if len(history_items) == 1:
        scores = score_single_item(history_items[0], history_set)
    else:
        scores = score_multi_item(history_items, history_set)
    
    for rank, item in enumerate(list(item_popularity.keys())[:100]):
        if item not in history_set:
            scores[item] += (100 - rank) * 0.05
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [item for item, _ in ranked[:top_n]]

print("Generating predictions...", flush=True)
test = pd.read_csv(f"{base_path}/test.csv")
results = []
popular = list(item_popularity.keys())[:100]

for idx, row in test.iterrows():
    hid = row['id']
    history = row['history'] if isinstance(row['history'], str) else ''
    history_items = [a for a in history.split() if a in meta_items]
    
    recs = ensemble_score(history_items, top_n=10)
    
    history_set = set(history_items)
    while len(recs) < 10:
        for p in popular:
            if p not in recs and p not in history_set:
                recs.append(p)
                if len(recs) >= 10:
                    break
    
    results.append([hid] + recs[:10])
    
    if idx % 500 == 0:
        print(f"Processed {idx}/{len(test)}", flush=True)

sub = pd.DataFrame(results, columns=['id'] + [f'rec{i}' for i in range(1, 11)])
sub.to_csv('/home/milkak/UniSRec/submission_brand_boost_only.csv', index=False)
print("\nSaved submission_brand_boost_only.csv", flush=True)

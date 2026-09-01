with open('scripts/evaluate_all.py', 'r') as f:
    content = f.read()

old_metrics_block = """                f1_score = metrics.get('f1', 0)
                auc_score = metrics.get('roc_auc', 0)
                
                results[os.path.basename(ckpt)] = f1_score
                print(f" -> F1 Score: {f1_score:.4f} | AUC: {auc_score:.4f}")"""

new_metrics_block = """                f1_score = metrics.get('f1', 0)
                auc_score = metrics.get('roc_auc', 0)
                precision = metrics.get('precision', 0)
                accuracy = metrics.get('accuracy', 0)
                
                results[os.path.basename(ckpt)] = f1_score
                print(f" -> F1: {f1_score:.4f} | AUC: {auc_score:.4f} | Precision: {precision:.4f} | Accuracy: {accuracy:.4f}")"""

content = content.replace(old_metrics_block, new_metrics_block)

with open('scripts/evaluate_all.py', 'w') as f:
    f.write(content)

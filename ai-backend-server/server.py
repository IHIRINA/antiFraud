import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import sys
import io

# 设置标准输出的编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning Rate')
# parser.add_argument('--learning_rate', default=1e-7, type=float, help='Learning Rate')
parser.add_argument('--warmup_proportion', default=0.05, type=float, help='Set Warmup Proportion')
parser.add_argument('--num_epochs', default=480, type=int, help='Set Number of Epochs')
parser.add_argument('--batch_size', default=1, type=int, help='Set Batch Size')
# parser.add_argument('--max_text_length', default=150, type=int, help='Set Max Text Length')
# parser.add_argument('--max_time', default=30, type=int, help='Set Max Time')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='Train Or Test Mode')
parser.add_argument('--gradient_accumulation_step', default=8, type=int) # 8

parser.add_argument('--model_output_dir', default='./', type=str)
# parser.add_argument('--acoustic_dim', default=100, type=int)
# parser.add_argument('--visual_dim', default=256, type=int)
# parser.add_argument('--text_dim', default=768, type=int)
# parser.add_argument('--beta_shift', default=1, type=int)
# parser.add_argument('--dropout_prob', default=0.5, type=int)

# parser.add_argument('--num_past_utts', default=1, type=int, help='过去的话语数量')
# parser.add_argument('--num_futu_utts', default=0, type=int, help='未来的话语数量')

# lineConGraph 定义 argparse 参数
parser.add_argument('--in_features', type=int, default=768, help='输入特征的维度')
parser.add_argument('--hidden_features', type=int, default=256, help='隐藏层特征的维度')
parser.add_argument('--out_features', type=int, default=4, help='输出特征的维度')
parser.add_argument('--num_heads', type=int, default=4, help='GATConv 中的注意力头数')
parser.add_argument('--momentum', type=float, default=0.6, help='SGD优化器的动量参数')
parser.add_argument('--dataset', type=str, default='SCAM', choices=['SCAM'], help='Dataset to use SCAM')
parser.add_argument('--gamma', type=int, default=4, help='focalloss的gamma')
args = parser.parse_args()


class lineConGraph(nn.Module):
    def __init__(self, args):
        super(lineConGraph, self).__init__()
        self.conv1 = GATConv(args.in_features, args.hidden_features, heads=args.num_heads)
        self.conv2 = GATConv(args.hidden_features * args.num_heads, args.out_features, heads=args.num_heads)
        self.fc_out = nn.Linear(args.out_features * args.num_heads, args.out_features)

    def forward(self, x, edge_index_other, edge_idx_speaker):
        x1 = F.elu(self.conv1(x, edge_index_other))
        x1 = F.elu(self.conv2(x1, edge_index_other))

        x2 = F.elu(self.conv1(x, edge_idx_speaker))
        x2 = F.elu(self.conv2(x2, edge_idx_speaker))

        x = x1 + x2
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        return x


model = lineConGraph(args)


checkpoint_path = './7682.ckt'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")
feature_extractor = AutoModelForSequenceClassification.from_pretrained("./bert-base-chinese", output_hidden_states=True).to(device)
from flask import Flask,request

app = Flask(__name__)

@app.route("/",methods=['POST'])
def run_ai():
    inputs = tokenizer(request.form["sentence"], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = feature_extractor(**inputs)
    features = outputs.hidden_states
    featurestensor = torch.mean(features[-1].detach(), dim=1).cpu()
    edge_idx_other = torch.tensor([[0], [0]], dtype=torch.long)
    edge_idx_speaker = torch.tensor([[0], [0]], dtype=torch.long)
    sentence_vectors = featurestensor.to(device)
    edge_idx_other = edge_idx_other.to(device)
    edge_idx_speaker = edge_idx_speaker.to(device)
    output = model(sentence_vectors, edge_idx_other, edge_idx_speaker)
    predicted_labels = torch.argmax(output, dim=1)
    sentiment_map = {
        0: '中性',
        1: '网络交易及兼职诈骗',
        2: '虚假金融及投资诈骗',
        3: '身份冒充及威胁诈骗'
    }
    predicted_category = sentiment_map.get(predicted_labels.item(), "未分类")
    return predicted_category
app.run(host='0.0.0.0',port='6666')
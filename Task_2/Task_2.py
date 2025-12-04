import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet, Food101
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, f1_score, davies_bouldin_score, classification_report
import os
from tqdm import tqdm
import warnings


# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore')

def main():
    # Check GPU availability
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Great! GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected, running on CPU...")

    # Create output directories if they don't exist
    # 如果输出目录不存在，则创建它们
    os.makedirs('output', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # ============ Part 1: Load Datasets (加载数据集) ============
    print("\n=== Loading Datasets (加载数据集) ===")

    # Image Preprocessing configuration
    # 图像预处理配置
    trans = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (统一图像尺寸)
        transforms.ToTensor(),          # Convert to Tensor (转换为 Tensor)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization (ImageNet 标准化)
                           std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset 1: Oxford Pets ---
    # Dataset URL: https://www.robots.ox.ac.uk/~vgg/data/pets/
    print("1. Downloading Oxford Pets... ")#(正在下载 Oxford Pets...)
    pets = OxfordIIITPet(root='./data', split='trainval',
                        download=True, transform=trans)

    # Setup DataLoader
    # 设置数据加载器
    bs = 64 if device.type == 'cuda' else 32
    pets_loader = DataLoader(pets, batch_size=bs, shuffle=False, num_workers=0)
    print(f"Oxford Pets: {len(pets)} images, {len(pets.classes)} classes (类别)")

    # --- Dataset 2: Food101 (Subset) ---
    # Dataset URL: https://www.kaggle.com/datasets/dansbecker/food-101
    print("\n2. Preparing Food101 Dataset (Subset of top 10 classes)...")
    #(正在准备 Food101 数据集 - 前10类子集...)

    full_food = Food101(root='./data', split='train', download=True, transform=trans)

    # Filter top 10 classes to reduce dataset size for efficiency
    # 筛选前 10 个类别以减小数据集大小，提高效率
    NUM_FOOD_CLASSES = 10
    # Get indices for the first 10 classes
    # 获取前 10 个类别的索引
    food_indices = [i for i, label in enumerate(full_food._labels) if label < NUM_FOOD_CLASSES]

    # Create a subset
    # 创建子集
    food_subset = Subset(full_food, food_indices)
    food_classes = full_food.classes[:NUM_FOOD_CLASSES]

    food_loader = DataLoader(food_subset, batch_size=bs, shuffle=False, num_workers=0)
    print(f"Food101 (Subset): {len(food_subset)} images, {NUM_FOOD_CLASSES} classes")
    print(f"Classes (类别): {food_classes}")

    # ============ Part 2: Load Models (加载模型) ============
    print("\n=== Loading Pre-trained Models  ===")#(加载预训练模型)

    # Load VGG16 model
    # 加载 VGG16 模型
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Remove the classifier layer (keep only feature extractor)
    # 移除分类层（只保留特征提取器）
    vgg_feat = nn.Sequential(*list(vgg.features.children()))
    vgg_feat = vgg_feat.to(device)
    vgg_feat.eval()  # Set to evaluation mode (设置为评估模式)
    print("✓ VGG16 loaded ")

    # Load ResNet50 model
    # 加载 ResNet50 模型
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final fully connected layer
    # 移除最后的全连接层
    resnet_feat = nn.Sequential(*list(resnet.children())[:-1])
    resnet_feat = resnet_feat.to(device)
    resnet_feat.eval()
    print("✓ ResNet50 loaded ")

    # ============ Part 3: Feature Extraction (特征提取) ============
    def get_features(loader, model, model_name):
        """
        Extract features using the specified model.
        使用指定模型提取特征。
        """
        feats = []
        lbls = []

        print(f"\nExtracting features using {model_name}")#(正在使用 {model_name} 提取特征...)
        with torch.no_grad():  # Disable gradient calculation (禁用梯度计算)
            for imgs, labels in tqdm(loader):
                imgs = imgs.to(device)

                if model_name == 'VGG':
                    f = model(imgs)
                    f = nn.functional.adaptive_avg_pool2d(f, (1, 1)) # Pooling (池化)
                    f = f.view(f.size(0), -1) # Flatten (展平)
                else:  # ResNet
                    f = model(imgs)
                    f = f.view(f.size(0), -1)

                feats.append(f.cpu().numpy()) # Move back to CPU (移回 CPU)
                lbls.extend(labels.numpy())

        feats = np.vstack(feats)
        lbls = np.array(lbls)
        print(f"Feature dimensions: {feats.shape}")# (特征维度)
        return feats, lbls

    # Perform Feature Extraction
    # 执行特征提取
    print("\nStarting feature extraction")

    # 1. Extract features from Pets dataset using VGG
    # 1. 使用 VGG 从 Pets 数据集提取特征
    pets_vgg_f, pets_l = get_features(pets_loader, vgg_feat, 'VGG')

    # 2. Extract features from Food dataset using ResNet
    # 2. 使用 ResNet 从 Food 数据集提取特征
    food_res_f, food_l = get_features(food_loader, resnet_feat, 'ResNet')

    # Save extracted features to disk
    # 将提取的特征保存到磁盘
    np.save('output/pets_vgg.npy', pets_vgg_f)
    np.save('output/food_resnet.npy', food_res_f)
    print("Features saved! ")#(特征已保存!)

    # ============ Part 4: Visualization (降维可视化) ============
    print("\n=== Dimensionality Reduction & Visualization  ===")#(降维与可视化)

    def visualize(feats, labels, name, n_classes):
        """
        Visualize high-dimensional features using PCA, t-SNE, and UMAP.
        使用 PCA、t-SNE 和 UMAP 可视化高维特征。
        """
        # Subsample data for faster visualization
        # 为了加快可视化速度，对数据进行采样
        n = min(1500, len(feats))
        idx = np.random.choice(len(feats), n, replace=False)
        f_sub = feats[idx]
        l_sub = labels[idx]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        cmap = 'tab10' if n_classes <= 10 else 'tab20' # Select colormap (选择颜色映射)

        # 1. PCA Visualization
        print(f"PCA on {name}...")
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(f_sub)
        ax[0].scatter(pca_res[:, 0], pca_res[:, 1], c=l_sub,
                     cmap=cmap, alpha=0.6, s=20)
        ax[0].set_title(f'PCA - {name}')

        # 2. t-SNE Visualization
        print(f"t-SNE on {name}...")
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_res = tsne.fit_transform(f_sub)
        ax[1].scatter(tsne_res[:, 0], tsne_res[:, 1], c=l_sub,
                     cmap=cmap, alpha=0.6, s=20)
        ax[1].set_title(f't-SNE - {name}')

        # 3. UMAP Visualization
        print(f"UMAP on {name}...")
        um = umap.UMAP(n_neighbors=15, min_dist=0.1)
        umap_res = um.fit_transform(f_sub)
        sc = ax[2].scatter(umap_res[:, 0], umap_res[:, 1], c=l_sub,
                          cmap=cmap, alpha=0.6, s=20)
        ax[2].set_title(f'UMAP - {name}')
        plt.colorbar(sc, ax=ax[2], label='Class Label')

        plt.tight_layout()
        plt.savefig(f'output/{name}_viz.png', dpi=100)
        plt.close()
        print(f"Image saved: {name}_viz.png (图片已保存)")

    # Run visualization for both datasets
    # 对两个数据集运行可视化
    visualize(pets_vgg_f, pets_l, 'Pets_VGG', 37)
    visualize(food_res_f, food_l, 'Food_ResNet', 10)

    # ============ Part 5: Clustering (聚类分析) ============
    print("\n=== Clustering Analysis ")

    def clustering(feats, name):
        """
        Perform K-means clustering and evaluate using Silhouette and Davies-Bouldin scores.
        执行 K-means 聚类并使用轮廓系数和 DB 指数进行评估。
        """
        # Reduce dimensions with PCA before clustering to improve performance
        # 聚类前使用 PCA 降维以提高性能
        pca = PCA(n_components=50)
        f_pca = pca.fit_transform(feats[:2000])  # Use subset for speed (使用子集加速)

        ks = range(2, 10)
        sil_scores = []
        db_scores = []

        print(f"Running clustering on {name}... ")
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            pred = km.fit_predict(f_pca)

            # Calculate metrics
            # 计算指标
            sil = silhouette_score(f_pca, pred)
            db = davies_bouldin_score(f_pca, pred)

            sil_scores.append(sil)
            db_scores.append(db)
            print(f"k={k}: Silhouette={sil:.3f}, DB_Score={db:.3f}")

        # Plot Metrics (Dual Axis)
        # 绘制指标图（双坐标轴）
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot Silhouette Score
        ax1.plot(ks, sil_scores, 'bo-', label='Silhouette (Higher is better)')
        ax1.set_xlabel('K (Number of Clusters)')
        ax1.set_ylabel('Silhouette Score', color='b')

        # Plot Davies-Bouldin Score
        ax2 = ax1.twinx()
        ax2.plot(ks, db_scores, 'rx--', label='DB Score (Lower is better)')
        ax2.set_ylabel('Davies-Bouldin Score', color='r')

        plt.title(f'{name} - Clustering Metrics')
        plt.savefig(f'output/{name}_cluster.png')
        plt.close()

        # Find best K based on Silhouette Score
        # 基于轮廓系数寻找最佳 K 值
        best_k = ks[np.argmax(sil_scores)]
        print(f"{name} Best k={best_k} (based on Silhouette)")
        return best_k

    # Run Clustering Analysis
    # 运行聚类分析
    clustering(pets_vgg_f, 'Pets')
    clustering(food_res_f, 'Food')

    # ============ Part 6: Classification (分类任务) ============
    print("\n=== Classification Task (on Food101) ===")

    # Define a Linear Classifier
    # 定义一个线性分类器
    class Classifier(nn.Module):  # [Modified] Class name changed from SimpleClassifier to Classifier
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.fc(x)

    # Prepare Classification Data
    # 准备分类数据
    print("\nTraining Food101 Classifier (10 classes)... ")

    # Split data into Training and Testing sets (80% Train, 20% Test)
    # 将数据划分为训练集和测试集（80% 训练，20% 测试）
    n_train = int(len(food_res_f) * 0.8)
    train_f = food_res_f[:n_train]
    train_l = food_l[:n_train]
    test_f = food_res_f[n_train:]
    test_l = food_l[n_train:]

    # Convert numpy arrays to PyTorch Tensors and move to GPU
    # 将 numpy 数组转换为 PyTorch 张量并移动到 GPU
    train_f_t = torch.FloatTensor(train_f).to(device)
    train_l_t = torch.LongTensor(train_l).to(device)
    test_f_t = torch.FloatTensor(test_f).to(device)
    # test_l stays numpy for sklearn metrics evaluation
    # test_l 保留为 numpy 格式用于 sklearn 指标评估

    # Initialize Classifier, Loss Function, and Optimizer
    # 初始化分类器、损失函数和优化器
    # [Modified] Instantiating Classifier instead of SimpleClassifier
    clf = Classifier(train_f.shape[1], NUM_FOOD_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

    # Training Loop
    # 训练循环
    print("Start Training... ")
    epochs = 20
    batch_size = 128

    clf.train()
    for ep in range(epochs):
        # Shuffle training data
        # 打乱训练数据
        perm = torch.randperm(len(train_f_t))
        train_f_t = train_f_t[perm]
        train_l_t = train_l_t[perm]

        losses = []
        for i in range(0, len(train_f_t), batch_size):
            batch_f = train_f_t[i:i+batch_size]
            batch_l = train_l_t[i:i+batch_size]

            # Forward pass
            # 前向传播
            out = clf(batch_f)
            loss = criterion(out, batch_l)

            # Backward pass and optimization
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {np.mean(losses):.4f}")

    # Testing Phase
    # 测试阶段
    print("\nTesting Classifier... ")
    clf.eval()
    with torch.no_grad():
        logits = clf(test_f_t)
        pred = logits.argmax(dim=1).cpu().numpy()

        # Calculate metrics
        # 计算指标
        acc = accuracy_score(test_l, pred)
        f1 = f1_score(test_l, pred, average='weighted')
        cm = confusion_matrix(test_l, pred)

        print(f"Accuracy: {acc:.3f}")
        print(f"F1-Score: {f1:.3f}")

        print("\nClassification Report ")
        # Use English class names for report
        print(classification_report(test_l, pred, target_names=food_classes))

        print(f"Confusion Matrix \n{cm}")

    print("\n=== Done! All results saved to 'output/' folder. ===")


if __name__ == '__main__':
    main()
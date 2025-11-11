import torch
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(fc_history, cnn_history):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def inference_time(model, test_loader, device="cuda", num_runs=10):
    model.to(device)
    model.eval()
    
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(images)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(images)
    end_time = time.time()
    
    avg_time_per_batch = (end_time - start_time) / num_runs
    return avg_time_per_batch



def plot_confusion_matrix_for_models(models, model_names, test_loader):
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 5))
    axes = axes.flatten() 
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        all_predictions = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to('cuda')
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   ax=axes[idx],
                   cbar_kws={'shrink': 0.8})
        
        accuracy = np.trace(cm) / np.sum(cm)
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()



def visualize_all_models_activations(models, model_names, test_loader, num_filters_to_show=7):
    
    images, labels = next(iter(test_loader))
    single_image = images[0:1]
    
    num_models = len(models)
    num_filters_to_show = num_filters_to_show
    
    fig, axes = plt.subplots(num_models, num_filters_to_show + 1, figsize=(18, 3 * num_models))
    
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    for model_idx, (model, name) in enumerate(zip(models, model_names)):
        model.eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    activations = torch.relu(module(single_image.to('cuda')))
                    break
        
        activations = activations[0].cpu().numpy()  # type: ignore # [32, 28, 28]
        
        axes[model_idx, 0].imshow(single_image[0][0], cmap='gray')
        axes[model_idx, 0].set_title(f'{name}\nИсходная')
        axes[model_idx, 0].axis('off')
        
        for filter_idx in range(num_filters_to_show):
            axes[model_idx, filter_idx + 1].imshow(activations[filter_idx], cmap='hot')
            axes[model_idx, filter_idx + 1].set_title(f'Ф{filter_idx}')
            axes[model_idx, filter_idx + 1].axis('off')
    
    plt.suptitle('Активации первого слоя всех моделей', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_all_models_feature_maps(models, model_names, test_loader):
    
    images, labels = next(iter(test_loader))
    single_image = images[0:1]
    
    num_models = len(models)
    num_filters_per_model = 6  
    
    fig, axes = plt.subplots(num_models, num_filters_per_model + 1, figsize=(18, 3 * num_models))
    
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    for model_idx, (model, name) in enumerate(zip(models, model_names)):
        model.eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    feature_maps = torch.relu(module(single_image.to('cuda')))
                    break
        
        feature_maps = feature_maps[0].cpu().numpy()
        
        axes[model_idx, 0].imshow(single_image[0][0], cmap='gray')
        axes[model_idx, 0].set_title(f'{name}\nИсходная')
        axes[model_idx, 0].axis('off')
        
        for filter_idx in range(num_filters_per_model):
            if filter_idx < feature_maps.shape[0]:
                axes[model_idx, filter_idx + 1].imshow(feature_maps[filter_idx], cmap='hot')
                axes[model_idx, filter_idx + 1].set_title(f'Ф{filter_idx}')
            axes[model_idx, filter_idx + 1].axis('off')
    
    plt.suptitle('Feature Maps первого слоя всех моделей', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


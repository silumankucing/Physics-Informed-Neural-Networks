import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# ========================================
# 1. Persiapan Data Simulasi
# ========================================

def generate_synthetic_data(num_samples=1000):
    
    """Generate synthetic training data for battery degradation"""
    np.random.seed(42)
    
    # Parameter simulasi
    time = np.linspace(0, 1000, num_samples)  # 0-1000 jam
    current = np.random.normal(1.0, 0.2, num_samples)  # Arus (A)
    temperature = np.random.normal(300, 10, num_samples)  # Suhu (K)
    
    # Simulasi degradasi kapasitas (ground truth)
    k = 0.001  # Koefisien degradasi
    capacity = np.exp(-k * time * (current**2) * np.exp(-1000/temperature))
    
    # Tambahkan noise
    capacity += np.random.normal(0, 0.01, num_samples)
    capacity = np.clip(capacity, 0, 1)
    
    return np.column_stack([current, temperature, time]), capacity

# Generate data
X_train, y_train = generate_synthetic_data()

# ========================================
# 2. Membangun Model PINN
# ========================================

def build_pinn_model():

    # [current, temperature, time]
    inputs = Input(shape=(3,))  
    
    # Hidden layers
    x = Dense(64, activation='tanh')(inputs)
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_pinn_model()

# ========================================
# 3. Loss Function Fisika + Data
# ========================================

def physics_loss(y_true, y_pred, X):
    """Physics-based loss for battery degradation"""
    current, temperature, time = X[:, 0], X[:, 1], X[:, 2]
    
    with tf.GradientTape() as tape:
        tape.watch(time)
        Q = model(X)
    dQ_dt = tape.gradient(Q, time)
    
    # Persamaan degradasi (simplified)
    k = 0.001
    physics_constraint = dQ_dt + k * (current**2) * Q * tf.exp(-1000/temperature)
    
    return tf.reduce_mean(physics_constraint**2)

def total_loss(y_true, y_pred, X):
    data_loss = tf.reduce_mean((y_true - y_pred)**2)
    pde_loss = physics_loss(y_true, y_pred, X)
    return data_loss + 0.1 * pde_loss  # Bobot untuk loss fisika

# ========================================
# 4. Kompilasi dan Pelatihan
# ========================================
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=lambda y_true, y_pred: total_loss(y_true, y_pred, X_train))

history = model.fit(X_train, y_train,
                   epochs=500,
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# ========================================
# 5. Visualisasi Hasil
# ========================================

def plot_results():
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot prediksi vs waktu
    test_time = np.linspace(0, 1000, 100)
    test_current = np.full_like(test_time, 1.0)  # Arus konstan 1A
    test_temp = np.full_like(test_time, 300)     # Suhu konstan 300K
    X_test = np.column_stack([test_current, test_temp, test_time])
    
    pred_capacity = model.predict(X_test)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_time, pred_capacity, 'r-', label='Prediksi PINN')
    plt.scatter(X_train[:, 2], y_train, s=5, alpha=0.3, label='Data Training')
    plt.xlabel('Waktu (jam)')
    plt.ylabel('Kapasitas Relatif')
    plt.legend()
    plt.tight_layout()
    plt.savefig('battery_degradation.png', dpi=300)
    plt.show()

plot_results()

# ========================================
# 6. Simpan Model dan Hasil
# ========================================
# Simpan model
model.save('pinn_battery_model.h5')

# Simpan prediksi
test_data = np.column_stack([
    np.linspace(0.5, 1.5, 5),  # Variasi arus
    np.linspace(290, 310, 5),   # Variasi suhu
    np.full(5, 500)             # Waktu tetap 500 jam
])
predictions = model.predict(test_data)

results = pd.DataFrame({
    'Current (A)': test_data[:, 0],
    'Temperature (K)': test_data[:, 1],
    'Time (hours)': test_data[:, 2],
    'Predicted Capacity': predictions.flatten()
})
results.to_csv('battery_predictions.csv', index=False)

print("Model dan hasil prediksi berhasil disimpan!")
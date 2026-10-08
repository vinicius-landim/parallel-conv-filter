import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configurações gerais dos experimentos
config_threads = [1, 2, 3, 4, 5, 6]

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_4K_DIR = PROJECT_ROOT / "data" / "5_4k_3840x2160"

# Executáveis
exec_serial = SRC_DIR / "conv_filter_serial.exe"
exec_omp = SRC_DIR / "openmp" / "conv_filter.exe"
exec_cuda = {
    "CUDA_GLOBAL": SRC_DIR / "cuda" / "conv_filter.exe",
    "CUDA_CONST": SRC_DIR / "cuda" / "conv_filter_constmem.exe",
    "CUDA_TILED": SRC_DIR / "cuda" / "conv_filter_tile.exe"
}

largura = 3840
altura = 2160
kernel_size = 31

pasta_graficos = SRC_DIR / "graficos"
pasta_tabelas = SRC_DIR / "tabelas"
pasta_graficos.mkdir(exist_ok=True)
pasta_tabelas.mkdir(exist_ok=True)

print("A utilizar os executáveis já compilados...\n")

env = os.environ.copy()
env["OMP_PROC_BIND"] = "close"
env["OMP_PLACES"] = "cores"

imagens_4k = sorted(list(DATA_4K_DIR.glob("*.png")))
num_imagens = len(imagens_4k)
print(f"Encontradas {num_imagens} imagens na pasta 4K.")

# ==========================================
# 1. EXECUÇÃO SERIAL
# ==========================================
print("\nExecutando versão serial para recolher métricas")
resultados_serial = []

for i, img_path in enumerate(imagens_4k):
    processo = subprocess.run([str(exec_serial), str(img_path)], capture_output=True, text=True, env=env, cwd=SRC_DIR)
    linhas = processo.stdout.strip().split('\n')
    encontrou = False
    for linha in reversed(linhas):
        saida = linha.strip().split(',')
        if len(saida) == 3 and saida[0] == "SERIAL":
            _, tempo, energia = saida
            resultados_serial.append({"Imagem": img_path.name, "Tempo (s)": float(tempo), "Energia (J)": float(energia)})
            encontrou = True
            break
    if not encontrou: print(f"Erro C: {processo.stderr.strip()}")

df_serial_raw = pd.DataFrame(resultados_serial)
tempo_serial_mediano = df_serial_raw["Tempo (s)"].median()
energia_serial_mediana = df_serial_raw["Energia (J)"].median()

# ==========================================
# 2. EXECUÇÃO OPENMP
# ==========================================
resultados_omp = []
for threads in config_threads:
    print(f"Executando OpenMP com {threads} thread(s)")
    for i, img_path in enumerate(imagens_4k):
        processo = subprocess.run([str(exec_omp), str(img_path), str(threads)], capture_output=True, text=True, env=env, cwd=SRC_DIR)
        linhas = processo.stdout.strip().split('\n')
        encontrou = False
        for linha in reversed(linhas):
            saida = linha.strip().split(',')
            if len(saida) == 3 and saida[0].isdigit():
                t, tempo, energia = saida
                resultados_omp.append({"Imagem": img_path.name, "Threads": int(t), "Tempo (s)": float(tempo), "Energia (J)": float(energia)})
                encontrou = True
                break

df_omp_raw = pd.DataFrame(resultados_omp)

# ==========================================
# 3. EXECUÇÃO CUDA
# ==========================================
resultados_cuda = []
for tag, exe in exec_cuda.items():
    print(f"Executando GPU ({tag})")
    for i, img_path in enumerate(imagens_4k):
        processo = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, env=env, cwd=SRC_DIR)
        linhas = processo.stdout.strip().split('\n')
        encontrou = False
        for linha in reversed(linhas):
            saida = linha.strip().split(',')
            if len(saida) == 3 and saida[0] == tag:
                _, tempo, energia = saida
                resultados_cuda.append({"Config": tag, "Imagem": img_path.name, "Tempo (s)": float(tempo), "Energia (J)": float(energia)})
                encontrou = True
                break

df_cuda_raw = pd.DataFrame(resultados_cuda)

# ==========================================
# 4. CÁLCULO DE MÉTRICAS (OPENMP & CUDA)
# ==========================================
# OpenMP
df_metrics = df_omp_raw.groupby("Threads").agg(
    Tempo_Medio=("Tempo (s)", "mean"), Tempo_Mediana=("Tempo (s)", "median"), Tempo_Desvio=("Tempo (s)", "std"),
    Energia_Media=("Energia (J)", "mean"), Energia_Mediana=("Energia (J)", "median"), Energia_Desvio=("Energia (J)", "std")
).reset_index()

df_metrics["Speedup"] = tempo_serial_mediano / df_metrics["Tempo_Mediana"]
df_metrics["Eficiencia_Paralela"] = df_metrics["Speedup"] / df_metrics["Threads"]
df_metrics["EDP (J.s)"] = df_metrics["Tempo_Mediana"] * df_metrics["Energia_Mediana"]
df_metrics["Potencia_Media_W"] = df_metrics["Energia_Mediana"] / df_metrics["Tempo_Mediana"]

flops_por_pixel = (kernel_size * kernel_size) * 2
total_flops = largura * altura * flops_por_pixel
total_megapixels = (largura * altura) / 1_000_000

df_metrics["MFLOPS"] = (total_flops / df_metrics["Tempo_Mediana"]) / 1_000_000
df_metrics["MFLOPS/Watt"] = df_metrics["MFLOPS"] / df_metrics["Potencia_Media_W"]
df_metrics["Megapixels/s"] = total_megapixels / df_metrics["Tempo_Mediana"]

# CUDA
df_cuda_metrics = df_cuda_raw.groupby("Config").agg(
    Tempo_Mediana=("Tempo (s)", "median"), Energia_Mediana=("Energia (J)", "median")
).reset_index()

tiled_tempo = df_cuda_metrics[df_cuda_metrics["Config"] == "CUDA_TILED"]["Tempo_Mediana"].values[0]
tiled_energia = df_cuda_metrics[df_cuda_metrics["Config"] == "CUDA_TILED"]["Energia_Mediana"].values[0]

tiled_speedup = tempo_serial_mediano / tiled_tempo
tiled_edp = tiled_tempo * tiled_energia
tiled_potencia = tiled_energia / tiled_tempo
tiled_mflops_w = ((total_flops / tiled_tempo) / 1_000_000) / tiled_potencia
tiled_throughput = total_megapixels / tiled_tempo

# CSVs
df_metrics.to_csv(pasta_tabelas / "resultados_metricas_cpu.csv", index=False)
df_cuda_metrics.to_csv(pasta_tabelas / "resultados_metricas_gpu.csv", index=False)

# ==========================================
# 5. GERAÇÃO DOS 10 GRÁFICOS
# ==========================================
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

threads_vals = df_metrics["Threads"].values
cor_cuda = "#2ca02c" # Verde para destacar a GPU

def salvar(fig, nome):
    fig.savefig(pasta_graficos / f"{nome}.png", dpi=300)
    fig.savefig(pasta_graficos / f"{nome}.pdf")
    plt.close(fig)

# G1: Tempo
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(threads_vals, df_metrics["Tempo_Mediana"], yerr=df_metrics["Tempo_Desvio"], marker='o', capsize=5, linewidth=2, label="OpenMP", color="#1f77b4")
ax.axhline(tempo_serial_mediano, color="#d62728", linestyle="--", linewidth=2, label=f"Serial ({tempo_serial_mediano:.2f}s)")
ax.axhline(tiled_tempo, color=cor_cuda, linestyle="-.", linewidth=2, label=f"CUDA Tiled ({tiled_tempo:.3f}s)")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Tempo de execução (s)")
ax.set_title("Tempo de execução da convolução (4K)")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "01_tempo_execucao")

# G2: Speedup
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads_vals, df_metrics["Speedup"], marker='o', linewidth=2, label="OpenMP medido", color="#1f77b4")
ax.plot(threads_vals, threads_vals, linestyle="--", linewidth=2, label="Speedup ideal (linear)", color="gray")
ax.axhline(tiled_speedup, color=cor_cuda, linestyle="-.", linewidth=2, label=f"CUDA Tiled ({tiled_speedup:.1f}x)")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Speedup")
ax.set_title("Speedup em relação à versão sequencial")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "02_speedup")

# G3: Eficiência Paralela (Apenas CPU)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads_vals, df_metrics["Eficiencia_Paralela"], marker='s', linewidth=2, color="#ff7f0e")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=2, label="Eficiência ideal (100%)")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Eficiência paralela")
ax.set_title("Eficiência paralela por número de threads (CPU)")
ax.set_xticks(threads_vals)
ax.set_ylim(0, 1.15)
ax.legend()
salvar(fig, "03_eficiencia_paralela")

# G4: Energia
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(threads_vals, df_metrics["Energia_Mediana"], yerr=df_metrics["Energia_Desvio"], capsize=5, color="#9467bd", alpha=0.8, edgecolor="black")
ax.axhline(energia_serial_mediana, color="#d62728", linestyle="--", linewidth=2, label=f"Serial ({energia_serial_mediana:.1f}J)")
ax.axhline(tiled_energia, color=cor_cuda, linestyle="-.", linewidth=2, label=f"CUDA Tiled ({tiled_energia:.1f}J)")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Energia (J)")
ax.set_title("Consumo de energia da convolução (4K)")
ax.set_xticks(threads_vals)
ax.legend(loc='lower left')
salvar(fig, "04_energia")

# G5: EDP 
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads_vals, df_metrics["EDP (J.s)"], marker='D', linewidth=2, color="#8c564b", label="OpenMP")
ax.axhline(tiled_edp, color=cor_cuda, linestyle="-.", linewidth=2, label="CUDA Tiled")
ax.set_xlabel("Número de threads")
ax.set_ylabel("EDP (J·s)")
ax.set_title("Energy-Delay Product (4K)")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "05_edp")

# G6: Potência
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads_vals, df_metrics["Potencia_Media_W"], marker='^', linewidth=2, color="#e377c2", label="OpenMP")
ax.axhline(tiled_potencia, color=cor_cuda, linestyle="-.", linewidth=2, label="CUDA Tiled")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Potência média (W)")
ax.set_title("Potência média durante a convolução")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "06_potencia")

# G7: MFLOPS/Watt
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(threads_vals, df_metrics["MFLOPS/Watt"], marker='v', linewidth=2, color="#17becf", label="OpenMP")
ax.axhline(tiled_mflops_w, color=cor_cuda, linestyle="-.", linewidth=2, label="CUDA Tiled")
ax.set_xlabel("Número de threads")
ax.set_ylabel("MFLOPS/Watt")
ax.set_title("Eficiência energética (4K)")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "07_mflops_por_watt")

# G8: Throughput
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(threads_vals, df_metrics["Megapixels/s"], color="#bcbd22", alpha=0.8, edgecolor="black", label="OpenMP")
ax.axhline(tiled_throughput, color=cor_cuda, linestyle="-.", linewidth=2, label="CUDA Tiled")
ax.set_xlabel("Número de threads")
ax.set_ylabel("Megapixels processados / segundo")
ax.set_title("Throughput da convolução")
ax.set_xticks(threads_vals)
ax.legend()
salvar(fig, "08_throughput")

# G9: Boxplot unificado
fig, ax = plt.subplots(figsize=(8, 5))
dados_box = [df_omp_raw[df_omp_raw["Threads"] == t]["Tempo (s)"].values for t in threads_vals]
dados_box.append(df_cuda_raw[df_cuda_raw["Config"] == "CUDA_TILED"]["Tempo (s)"].values)
labels_box = [f"{t}T" for t in threads_vals] + ["CUDA\nTiled"]
ax.boxplot(dados_box, tick_labels=labels_box, showmeans=True)
ax.set_ylabel("Tempo de execução (s)")
ax.set_title("Distribuição dos tempos (Dataset 4K)")
salvar(fig, "09_boxplot_tempos")

# G10: Comparação das Técnicas CUDA (Micro)
fig, ax = plt.subplots(figsize=(7, 5))
tags_cuda = ["CUDA_GLOBAL", "CUDA_CONST", "CUDA_TILED"]
labels_cuda = ["Memória Global", "Mem. Constante", "Mem. Partilhada\n(Tiled)"]
tempos_cuda = [df_cuda_metrics[df_cuda_metrics["Config"] == t]["Tempo_Mediana"].values[0] for t in tags_cuda]

barras = ax.bar(labels_cuda, tempos_cuda, color=["#ff7f0e", "#1f77b4", cor_cuda], edgecolor="black", alpha=0.9)
ax.set_ylabel("Tempo de Execução (s)")
ax.set_title("Evolução das Otimizações na GPU (CUDA)")

for barra in barras:
    yval = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2.0, yval, f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')

salvar(fig, "10_comparacao_cuda")

print("\nProcessamento Finalizado! Os 10 gráficos foram exportados.")
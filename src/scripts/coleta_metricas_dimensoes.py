import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re

# Configurações gerais dos experimentos
config_threads = [1, 2, 3, 4, 5, 6]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
executavel_omp = PROJECT_ROOT / "src" / "openmp" / "conv_filter.exe"
executavel_serial = PROJECT_ROOT / "src" / "serial" / "conv_filter.exe"
DATA_DIR = PROJECT_ROOT / "data"

pasta_graficos = PROJECT_ROOT / "graficos"
pasta_tabelas = PROJECT_ROOT / "tabelas"
pasta_graficos.mkdir(exist_ok=True)
pasta_tabelas.mkdir(exist_ok=True)

# Fixa as threads (melhoria de estabilidade do SO)
env = os.environ.copy()
env["OMP_PROC_BIND"] = "close"
env["OMP_PLACES"] = "cores"

def extrair_dimensoes(nome_pasta):
    match = re.search(r"(\d+)x(\d+)", nome_pasta)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Não extraiu dimensões de: {nome_pasta}")


print("\n[1/2] Executando versão serial para as dimensões encontradas...")
resultados_serial = []

subpastas = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and re.search(r"\d+x\d+", p.name))

for subfolder in subpastas:
    resolucao_nome = subfolder.name
    largura, altura = extrair_dimensoes(resolucao_nome)
    imagens = sorted(list(subfolder.glob("*.png")))

    if not imagens:
        continue
        
    print(f" -> Processando {resolucao_nome} (Serial, {len(imagens)} imagens)...")

    for img_path in imagens:
        processo = subprocess.run([str(executavel_serial), str(img_path)], capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
        
        # Parser de baixo para cima para ignorar avisos do PCM
        linhas = processo.stdout.strip().split('\n')
        for linha in reversed(linhas):
            saida = linha.strip().split(',')
            if len(saida) == 3 and saida[0] == "SERIAL":
                _, tempo, energia = saida
                resultados_serial.append({
                    "Resolucao": resolucao_nome, "Largura": largura, "Altura": altura,
                    "Imagem": img_path.name,
                    "Tempo (s)": float(tempo), "Energia (J)": float(energia)
                })
                break

df_serial_raw = pd.DataFrame(resultados_serial)
# Calcula a mediana de tempo e energia usando todas as imagens da pasta
df_serial = df_serial_raw.groupby(["Resolucao", "Largura", "Altura"]).agg(
    Tempo_Serial=("Tempo (s)", "median"),
    Energia_Serial=("Energia (J)", "median")
).reset_index()



print("\n[2/2] Executando OpenMP...")
resultados_omp = []

for subfolder in subpastas:
    resolucao_nome = subfolder.name
    largura, altura = extrair_dimensoes(resolucao_nome)
    imagens = sorted(list(subfolder.glob("*.png")))

    if not imagens:
        continue

    for threads in config_threads:
        print(f" -> Processando {resolucao_nome} | {threads} Threads ({len(imagens)} imagens)...")

        for img_path in imagens:
            processo = subprocess.run([str(executavel_omp), str(img_path), str(threads)], capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
            
            linhas = processo.stdout.strip().split('\n')
            for linha in reversed(linhas):
                saida = linha.strip().split(',')
                if len(saida) == 3 and saida[0] == str(threads):
                    t, tempo, energia = saida
                    resultados_omp.append({
                        "Resolucao": resolucao_nome, "Largura": largura, "Altura": altura, 
                        "Imagem": img_path.name, "Threads": int(t),
                        "Tempo (s)": float(tempo), "Energia (J)": float(energia)
                    })
                    break

df_omp_raw = pd.DataFrame(resultados_omp)
# Calcula a mediana de tempo e energia usando todas as imagens da pasta
df_omp = df_omp_raw.groupby(["Resolucao", "Largura", "Altura", "Threads"]).agg(
    Tempo_OMP=("Tempo (s)", "median"),
    Energia_OMP=("Energia (J)", "median")
).reset_index()


df_metrics = df_omp.merge(df_serial[["Resolucao", "Tempo_Serial", "Energia_Serial"]], on="Resolucao")

df_metrics["Speedup"] = df_metrics["Tempo_Serial"] / df_metrics["Tempo_OMP"]
df_metrics["Eficiencia_Paralela"] = df_metrics["Speedup"] / df_metrics["Threads"]
df_metrics["EDP (J.s)"] = df_metrics["Tempo_OMP"] * df_metrics["Energia_OMP"]

df_metrics["Total_Megapixels"] = (df_metrics["Largura"] * df_metrics["Altura"]) / 1_000_000

# Ordenar logicamente pelo peso da imagem
df_metrics = df_metrics.sort_values(by=["Total_Megapixels", "Threads"])

# Renomear colunas para o output final
df_metrics.rename(columns={"Tempo_OMP": "Tempo (s)", "Energia_OMP": "Energia (J)"}, inplace=True)

print("\n--- Resumo Concluído ---")
df_resumo = df_metrics[["Resolucao", "Threads", "Tempo (s)", "Speedup", "Eficiencia_Paralela", "Energia (J)"]]
print(df_resumo.to_string(index=False))

# Exportar dados agregados e brutos
df_resumo.to_latex(pasta_tabelas / "tabela_resumo_dimensoes.tex", index=False, float_format="%.4f", caption="Métricas medianas por dimensão (20 imagens).", label="tab:dimensoes")
df_metrics.to_csv(pasta_tabelas / "resultados_finais_dimensoes.csv", index=False)
df_serial_raw.to_csv(pasta_tabelas / "raw_serial.csv", index=False)
df_omp_raw.to_csv(pasta_tabelas / "raw_openmp.csv", index=False)



plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
resolucoes_ordenadas = df_metrics["Resolucao"].unique()

def salvar(fig, nome):
    fig.savefig(pasta_graficos / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(pasta_graficos / f"{nome}.png", bbox_inches="tight")
    plt.close(fig)

# G1: Tempo
fig, ax = plt.subplots(figsize=(8, 5))
for res in resolucoes_ordenadas:
    sub = df_metrics[df_metrics["Resolucao"] == res]
    ax.plot(sub["Threads"], sub["Tempo (s)"], marker='o', linewidth=2, label=res)

ax.set_xlabel("Número de Threads")
ax.set_ylabel("Tempo Mediano (s)")
ax.set_title("Tempo de Execução por Resolução")
ax.set_xticks(config_threads)
ax.legend(title="Resolução")
salvar(fig, "01_tempo_resolucoes")

# G2: Speedup
fig, ax = plt.subplots(figsize=(8, 5))
for res in resolucoes_ordenadas:
    sub = df_metrics[df_metrics["Resolucao"] == res]
    ax.plot(sub["Threads"], sub["Speedup"], marker='o', linewidth=2, label=res)
ax.plot(config_threads, config_threads, linestyle="--", linewidth=2, label="Ideal", color="black")
ax.set_xlabel("Número de Threads")
ax.set_ylabel("Speedup")
ax.set_title("Escalabilidade por Carga de Trabalho")
ax.set_xticks(config_threads)
ax.legend(title="Resolução")
salvar(fig, "02_speedup_resolucoes")

# G3: Eficiência
fig, ax = plt.subplots(figsize=(8, 5))
for res in resolucoes_ordenadas:
    sub = df_metrics[df_metrics["Resolucao"] == res]
    ax.plot(sub["Threads"], sub["Eficiencia_Paralela"], marker='s', linewidth=2, label=res)
ax.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Ideal")
ax.set_xlabel("Número de Threads")
ax.set_ylabel("Eficiência Paralela")
ax.set_title("Eficiência do OpenMP por Resolução")
ax.set_xticks(config_threads)
ax.legend(title="Resolução")
salvar(fig, "03_eficiencia_resolucoes")

# G4: Energia
fig, ax = plt.subplots(figsize=(8, 5))
for res in resolucoes_ordenadas:
    sub = df_metrics[df_metrics["Resolucao"] == res]
    ax.plot(sub["Threads"], sub["Energia (J)"], marker='D', linewidth=2, label=res)

if df_metrics["Energia (J)"].min() > 0:
    ax.set_yscale('log')
    ax.set_ylabel("Energia Mediana (J) - Escala Log")
else:
    ax.set_ylabel("Energia Mediana (J)")

ax.set_xlabel("Número de Threads")
ax.set_title("Impacto Energético por Resolução")
ax.set_xticks(config_threads)
ax.legend(title="Resolução")
salvar(fig, "04_energia_resolucoes")

print("\nProcessamento em lote finalizado com sucesso!")
# parallel-conv-filter

Este repositório contém o Trabalho 1 da disciplina Programação Concorrente e Distribuída ministrada no 2° semestre de 2026 no ICT-Unifesp. 

O objetivo do trabalho é implementar a paralelização de um algoritmo de filtro convolucional utilizando a biblioteca OpenMP e a plataforma NVDIA CUDA, e fazer a comparação com a execução sequencial do algoritmo em termos de desempenho e eficiência energética. 

# Equipe
* [Enzo Belfort](https://github.com/EnzoBelfort)
* [Luiza de Souza](https://github.com/souza-luiza)
* [Natália Amaral](https://github.com/nataliavitoriaamaral)
* [Vinícius Landim](https://github.com/vinicius-landim)
* [Viviane Park](https://github.com/parkvivi)

# Estrutura do Repositório
```
parallel-conv-filter/
├── artigo/                     # arquivos LaTeX do artigo
├── data/                       # imagens teste
├── lib/
├── src/                        # algoritmo sequencial e paralelizado 
├── Makefile
└── README.md
```

# Automações
* Para compilar (versão sequencial e OpenMP):
```bash
make build
```

* Para executar testes e coletar dados:
```bash
make bench
```

* Para compilar e executar:
```bash
make all
```

* Para limpar arquivos:
```bash
make clean
```

# Efeito do Filtro😍😍
![original](data/output_files/bird-4k-gray.png)
![blur](data/output_files/bird-4k-blur.png)

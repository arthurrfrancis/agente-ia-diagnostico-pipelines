# Uso de um agente de IA baseado em LLM para diagnóstico de falhas em pipelines de dados

Código e dados do experimento do TCC II (Bacharelado em Engenharia de Software, Univille). O experimento compara um monitor de regras estáticas com um agente baseado em LLM (Gemini 3.8 Flash) no diagnóstico de causa raiz de falhas em um pipeline de dados sintético.

## Protocolo

Quatro cenários, dez execuções cada, com sementes distintas e ordem embaralhada (semente fixa). Cada execução tem cinco janelas normais seguidas de uma janela com a falha injetada.

| Cenário | Falha injetada na última janela | Causa raiz esperada |
|---|---|---|
| C0 – Controle | Nenhuma | Ausência de falha |
| C1 – Schema Drift | Esquema v1.2 → v1.3_incompativel; volume 0; erro 100% | Incompatibilidade de esquema |
| C2 – Interrupção da fonte | Volume 0; erro 0%; esquema v1.2 | Indisponibilidade da fonte |
| C3 – Falha silenciosa | Valor médio ×100 (centavos em vez de reais) | Desvio semântico de escala |

- **Linha de base:** alerta quando o volume é menor que 500 registros por janela ou a taxa de erro passa de 5%.
- **Agente:** um único agente com papel de SRE, temperatura 0,2 e resposta em JSON com esquema imposto (`falha_detectada`, `causa_raiz`, `grau_confianca_percentual`, `acao_corretiva_sugerida`). Ele recebe a série inteira e não depende de alerta prévio.
- **Métricas:** M1 detecção; M2 causa raiz (correta/parcial/incorreta); M3 ação corretiva (adequada/inadequada); M4 confiança autodeclarada comparada com M2; M5 latência.

## Como executar

Requer Python 3.13.

```bash
pip install -r requirements.txt
```

Teste do fluxo sem chamar a API (respostas simuladas):

```bash
python experimento.py rodar --seco
```

Execução real (requer uma chave da API do Gemini na variável de ambiente `GEMINI_API_KEY`):

```bash
python experimento.py rodar
```

Se a cota da API acabar no meio da rodada, retome com:

```bash
python experimento.py rodar --retomar resultados/<pasta>
```

Depois, preencha manualmente as colunas `m2_causa` e `m3_acao` de `resultados/<pasta>/classificacao.csv` (cenários C1 a C3) e gere as tabelas:

```bash
python experimento.py resumir resultados/<pasta>/classificacao.csv
```

## Resultados publicados

`resultados/20261018_200036/` contém a rodada reportada no artigo:

- `configuracao.json`: modelo, parâmetros, prompt e sessões de execução
- `execucoes.jsonl`: série enviada, resposta bruta do modelo, alertas das regras, latência e tokens de cada execução
- `classificacao.csv`: uma linha por execução, com a classificação manual de M2 e M3
- `log_retomada.txt`: log da segunda sessão (a rodada foi dividida em duas sessões por limite de cota, sem mudar parâmetros nem sementes)

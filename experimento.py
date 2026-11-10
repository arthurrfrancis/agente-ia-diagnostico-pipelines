"""
Experimento do TCC II - Diagnóstico de falhas em pipelines de dados com agente de IA.

Compara, nos mesmos dados, um monitor de regras estáticas e um agente baseado em LLM
(Gemini 3.8 Flash) em quatro cenários:
    C0 - Controle (sem falha)
    C1 - Schema Drift
    C2 - Interrupção da fonte de dados
    C3 - Falha silenciosa de qualidade (valor em centavos em vez de reais)

Uso:
    python experimento.py rodar              # executa o experimento com a API do Gemini
    python experimento.py rodar --seco       # testa o fluxo sem chamar a API (respostas falsas)
    python experimento.py rodar --retomar <pasta>  # continua uma execução interrompida (ex.: cota esgotada)
    python experimento.py resumir <arquivo>  # gera as Tabelas 1 e 2 a partir do CSV classificado

Requer a variável de ambiente GEMINI_API_KEY (exceto no modo --seco).
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuração do experimento (reportar estes valores na Seção 3 do artigo)
# ---------------------------------------------------------------------------
MODELO = "gemini-3.8-flash"  # gemini-2.5-flash (TCC I) descontinuado para novos usuários
TEMPERATURA = 0.2          # fixa; reportar no artigo
EXECUCOES_POR_CENARIO = 10
JANELAS_NORMAIS = 5        # janelas normais antes da janela final
PAUSA_ENTRE_CHAMADAS_S = 7 # evita estourar o limite de requisições por minuto
MAX_TENTATIVAS = 4

# Regras estáticas (linha de base)
VOLUME_MINIMO = 500        # registros por janela
TAXA_ERRO_MAXIMA = 5.0     # percentual

# Padrão "normal" do pipeline sintético
VOLUME_MEDIO, VOLUME_DESVIO = 1000, 40
ERRO_MIN, ERRO_MAX = 0.1, 1.0
VALOR_MEDIO_REAIS, VALOR_DESVIO = 150.0, 5.0
ESQUEMA_PADRAO = "v1.2"

CENARIOS = {
    "C0": {"nome": "Controle",
           "gabarito_causa": "Ausência de falha",
           "falha_esperada": False},
    "C1": {"nome": "Schema Drift",
           "gabarito_causa": "Incompatibilidade de esquema na origem (v1.2 -> v1.3_incompativel)",
           "falha_esperada": True},
    "C2": {"nome": "Interrupção da fonte",
           "gabarito_causa": "Indisponibilidade da fonte de dados (nenhum registro chega; esquema inalterado)",
           "falha_esperada": True},
    "C3": {"nome": "Falha silenciosa",
           "gabarito_causa": "Desvio semântico de escala: valores em centavos em vez de reais (x100)",
           "falha_esperada": True},
}


# ---------------------------------------------------------------------------
# Data Plane: geração das métricas sintéticas
# ---------------------------------------------------------------------------
def janela_normal(rng: random.Random, ts: datetime) -> dict:
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "volume_registros": max(0, int(rng.gauss(VOLUME_MEDIO, VOLUME_DESVIO))),
        "taxa_erros_percentual": round(rng.uniform(ERRO_MIN, ERRO_MAX), 2),
        "schema_versao": ESQUEMA_PADRAO,
        "valor_medio_transacao": round(rng.gauss(VALOR_MEDIO_REAIS, VALOR_DESVIO), 2),
    }


def gerar_serie(cenario: str, semente: int) -> list[dict]:
    rng = random.Random(semente)
    inicio = datetime(2026, 10, 20, 8, 0, 0) + timedelta(minutes=rng.randint(0, 600))
    serie = [janela_normal(rng, inicio + timedelta(minutes=i)) for i in range(JANELAS_NORMAIS + 1)]
    final = serie[-1]

    if cenario == "C1":    # Schema Drift: parser quebra, nada é processado
        final.update(volume_registros=0, taxa_erros_percentual=100.0,
                     schema_versao="v1.3_incompativel", valor_medio_transacao=None)
    elif cenario == "C2":  # Interrupção da fonte: nada chega, mas não há erro de parsing
        final.update(volume_registros=0, taxa_erros_percentual=0.0,
                     valor_medio_transacao=None)
    elif cenario == "C3":  # Falha silenciosa: tudo passa nas validações, valor fica x100
        final["valor_medio_transacao"] = round(final["valor_medio_transacao"] * 100, 2)
    # C0: série inteira normal
    return serie


# ---------------------------------------------------------------------------
# Linha de base: monitoramento por regras estáticas
# ---------------------------------------------------------------------------
def monitor_regras(serie: list[dict]) -> dict:
    violacoes = []
    for j in serie:
        if j["volume_registros"] < VOLUME_MINIMO:
            violacoes.append(f"{j['timestamp']}: volume {j['volume_registros']} < {VOLUME_MINIMO}")
        if j["taxa_erros_percentual"] > TAXA_ERRO_MAXIMA:
            violacoes.append(f"{j['timestamp']}: erro {j['taxa_erros_percentual']}% > {TAXA_ERRO_MAXIMA}%")
    return {"alerta": bool(violacoes), "violacoes": violacoes}


# ---------------------------------------------------------------------------
# Agente de diagnóstico (LLM)
# ---------------------------------------------------------------------------
class Diagnostico(BaseModel):
    falha_detectada: bool
    causa_raiz: str
    grau_confianca_percentual: int
    acao_corretiva_sugerida: str


PROMPT_SISTEMA = (
    "Você é um Engenheiro de Confiabilidade de Dados (SRE) responsável por um pipeline "
    "de ingestão de transações. Analise a série temporal de métricas operacionais e diga "
    "se há falha no pipeline. Se houver, identifique a causa raiz mais provável e sugira "
    "uma ação corretiva. Se não houver falha, informe falha_detectada como false. "
    "Informe também seu grau de confiança no diagnóstico, de 0 a 100."
)

DESCRICAO_CAMPOS = (
    "Campos de cada janela: timestamp; volume_registros (registros processados na janela); "
    "taxa_erros_percentual (registros rejeitados pelo parser, em %); schema_versao (versão do "
    "esquema recebido da origem); valor_medio_transacao (valor médio das transações, em reais; "
    "null quando não há registros processados)."
)


def montar_prompt(serie: list[dict]) -> str:
    return f"{DESCRICAO_CAMPOS}\n\nSérie (ordem cronológica):\n{json.dumps(serie, ensure_ascii=False, indent=2)}"


class CotaEsgotada(Exception):
    """Limite de requisições da API persistiu após todas as tentativas (HTTP 429)."""


def _tokens(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    campos = {"entrada": "prompt_token_count", "saida": "candidates_token_count",
              "raciocinio": "thoughts_token_count", "total": "total_token_count"}
    return {k: (getattr(u, v, None) if u else None) for k, v in campos.items()}


def chamar_agente(cliente, serie: list[dict]) -> tuple[dict, str, float, dict]:
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=PROMPT_SISTEMA,
        temperature=TEMPERATURA,
        response_mime_type="application/json",
        response_schema=Diagnostico,
    )
    ultimo_erro = None
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            t0 = time.perf_counter()
            resp = cliente.models.generate_content(model=MODELO, contents=montar_prompt(serie), config=config)
            latencia = time.perf_counter() - t0
            texto = resp.text
            return Diagnostico.model_validate_json(texto).model_dump(), texto, latencia, _tokens(resp)
        except Exception as e:  # rede, limite de taxa, JSON inválido
            ultimo_erro = e
            espera = 10 * tentativa
            print(f"    tentativa {tentativa} falhou ({type(e).__name__}: {e}); aguardando {espera}s")
            time.sleep(espera)
    if isinstance(ultimo_erro, errors.APIError) and ultimo_erro.code == 429:
        raise CotaEsgotada(str(ultimo_erro))
    raise RuntimeError(f"Falha após {MAX_TENTATIVAS} tentativas: {ultimo_erro}")


def agente_seco(serie: list[dict], rng: random.Random) -> tuple[dict, str, float, dict]:
    """Resposta falsa para testar o fluxo sem gastar chamadas de API."""
    final = serie[-1]
    falha = final["volume_registros"] == 0 or final["schema_versao"] != ESQUEMA_PADRAO
    d = {"falha_detectada": falha,
         "causa_raiz": "[SECO] resposta simulada",
         "grau_confianca_percentual": rng.randint(70, 100),
         "acao_corretiva_sugerida": "[SECO] resposta simulada"}
    return d, json.dumps(d, ensure_ascii=False), 0.0, {}


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def linha_csv(registro: dict) -> dict:
    cen = registro["cenario"]
    ag = registro["agente"] or {}
    tk = registro.get("tokens") or {}
    return {
        "cenario": cen, "execucao": registro["execucao"], "semente": registro["semente"],
        "gabarito_causa": CENARIOS[cen]["gabarito_causa"],
        "regras_alerta": registro["regras"]["alerta"],
        "agente_falha_detectada": ag.get("falha_detectada"),
        "m1_deteccao_correta": (ag.get("falha_detectada") == CENARIOS[cen]["falha_esperada"]) if ag else None,
        "causa_raiz": ag.get("causa_raiz"),
        "acao_corretiva_sugerida": ag.get("acao_corretiva_sugerida"),
        "confianca": ag.get("grau_confianca_percentual"),
        "latencia_s": registro["latencia_s"],
        "tokens_entrada": tk.get("entrada"),
        "tokens_saida": tk.get("saida"),
        "tokens_raciocinio": tk.get("raciocinio"),
        "tokens_total": tk.get("total"),
        "erro_execucao": registro["erro"],
        "m2_causa": "",   # preencher: correta / parcial / incorreta (C0: deixar vazio)
        "m3_acao": "",    # preencher: adequada / inadequada (C0: deixar vazio)
    }


def rodar(seco: bool, retomar: str | None = None) -> None:
    config_exp = {"modelo": MODELO, "temperatura": TEMPERATURA, "execucoes_por_cenario": EXECUCOES_POR_CENARIO,
                  "janelas_normais": JANELAS_NORMAIS, "volume_minimo": VOLUME_MINIMO,
                  "taxa_erro_maxima": TAXA_ERRO_MAXIMA, "prompt_sistema": PROMPT_SISTEMA,
                  "descricao_campos": DESCRICAO_CAMPOS, "modo_seco": seco}

    if retomar:
        pasta = Path(retomar)
        anterior = json.loads((pasta / "configuracao.json").read_text(encoding="utf-8"))
        anterior.pop("sessoes", None)
        if anterior != config_exp:
            sys.exit(f"Configuração atual difere da gravada em {pasta}/configuracao.json; retomada cancelada.")
    else:
        pasta = Path("resultados") / datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta.mkdir(parents=True, exist_ok=True)
    arq_jsonl = pasta / "execucoes.jsonl"
    arq_csv = pasta / "classificacao.csv"
    arq_config = pasta / "configuracao.json"

    # Execuções já concluídas sem erro são mantidas; as com erro são refeitas
    registros = {}
    if arq_jsonl.exists():
        for l in arq_jsonl.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                registros[(r["cenario"], r["execucao"])] = r
    feitos = {k for k, r in registros.items() if not r["erro"]}

    sessoes = json.loads(arq_config.read_text(encoding="utf-8")).get("sessoes", []) if retomar else []
    sessoes.append({"inicio": datetime.now().isoformat(timespec="seconds"), "ja_concluidas": len(feitos)})
    arq_config.write_text(json.dumps({**config_exp, "sessoes": sessoes}, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    cliente = None
    if not seco:
        from google import genai
        cliente = genai.Client()  # lê GEMINI_API_KEY do ambiente

    # Plano: cada cenário x N execuções, em ordem embaralhada (semente fixa)
    plano = [(c, i) for c in CENARIOS for i in range(1, EXECUCOES_POR_CENARIO + 1)]
    random.Random(42).shuffle(plano)
    pendentes = [p for p in plano if p not in feitos]
    if feitos:
        print(f"Retomando {pasta}: {len(feitos)} execuções já concluídas, {len(pendentes)} pendentes.")

    rng_seco = random.Random(7)
    interrompido = False
    for n, (cen, i) in enumerate(pendentes, 1):
        semente = 1000 * (list(CENARIOS).index(cen) + 1) + i
        serie = gerar_serie(cen, semente)
        regras = monitor_regras(serie)
        print(f"[{len(feitos) + n}/{len(plano)}] {cen} execução {i} (semente {semente})")

        registro = {"cenario": cen, "execucao": i, "semente": semente, "serie": serie, "regras": regras}
        try:
            if seco:
                diag, bruto, lat, tokens = agente_seco(serie, rng_seco)
            else:
                diag, bruto, lat, tokens = chamar_agente(cliente, serie)
            registro.update(agente=diag, resposta_bruta=bruto, latencia_s=round(lat, 3), tokens=tokens, erro=None)
        except CotaEsgotada as e:
            print(f"\nCota da API esgotada; execução interrompida sem registrar esta tentativa.\n  {e}")
            print(f"Para continuar depois: python experimento.py rodar --retomar {pasta}")
            interrompido = True
            break
        except Exception as e:
            registro.update(agente=None, resposta_bruta=None, latencia_s=None, tokens=None, erro=str(e))
            print(f"    ERRO registrado: {e}")

        registros[(cen, i)] = registro
        with arq_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        if not seco and n < len(pendentes):
            time.sleep(PAUSA_ENTRE_CHAMADAS_S)

    linhas = [linha_csv(r) for r in sorted(registros.values(), key=lambda r: (r["cenario"], r["execucao"]))]
    if linhas:
        with arq_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(linhas)

    if interrompido:
        print(f"\n{len(linhas)}/{len(plano)} execuções gravadas em {pasta}/ (incompleto).")
        return
    print(f"\nConcluído. Arquivos em {pasta}/")
    print("Próximo passo: preencher as colunas m2_causa e m3_acao em classificacao.csv e rodar:")
    print(f"  python experimento.py resumir {arq_csv}")


# ---------------------------------------------------------------------------
# Resumo: Tabelas 1 e 2 do artigo
# ---------------------------------------------------------------------------
def _bool(v: str):
    return {"True": True, "False": False}.get(v)


def _media(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else "-"


def resumir(caminho: str) -> None:
    with open(caminho, encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))

    pendentes = [l for l in linhas if l["cenario"] != "C0" and not l["erro_execucao"]
                 and (not l["m2_causa"].strip() or not l["m3_acao"].strip())]
    if pendentes:
        print(f"Atenção: {len(pendentes)} linhas sem classificação M2/M3; serão ignoradas nessas métricas.\n")

    print("TABELA 1 - Resultados por cenário")
    print(f"{'Cenário':<8}{'Regras alerta':>14}{'Agente M1':>11}{'Causa ok M2':>13}{'Ação ok M3':>12}"
          f"{'Conf. média':>13}{'Lat. média s':>14}{'Erros exec.':>13}")
    for cen in CENARIOS:
        ls = [l for l in linhas if l["cenario"] == cen]
        ok = [l for l in ls if not l["erro_execucao"]]
        n = len(ls)
        regras = sum(_bool(l["regras_alerta"]) is True for l in ls)
        m1 = sum(_bool(l["m1_deteccao_correta"]) is True for l in ok)
        if cen == "C0":
            m2 = m3 = "-"
        else:
            m2 = f"{sum(l['m2_causa'].strip().lower() == 'correta' for l in ok)}/{n}"
            m3 = f"{sum(l['m3_acao'].strip().lower() == 'adequada' for l in ok)}/{n}"
        conf = _media([float(l["confianca"]) for l in ok if l["confianca"]])
        lat = _media([float(l["latencia_s"]) for l in ok if l["latencia_s"]])
        print(f"{cen:<8}{f'{regras}/{n}':>14}{f'{m1}/{n}':>11}{m2:>13}{m3:>12}{conf:>13}{lat:>14}{n - len(ok):>13}")

    print("\nTABELA 2 - Confiança segundo a classificação da causa raiz")
    print(f"{'Cenário':<8}{'Correta (n)':>13}{'Conf. média':>13}{'Parcial/incorreta (n)':>23}{'Conf. média':>13}")
    for cen in ("C1", "C2", "C3"):
        ok = [l for l in linhas if l["cenario"] == cen and not l["erro_execucao"] and l["confianca"]]
        cor = [float(l["confianca"]) for l in ok if l["m2_causa"].strip().lower() == "correta"]
        nao = [float(l["confianca"]) for l in ok if l["m2_causa"].strip().lower() in ("parcial", "incorreta")]
        print(f"{cen:<8}{len(cor):>13}{_media(cor):>13}{len(nao):>23}{_media(nao):>13}")

    c0 = [l for l in linhas if l["cenario"] == "C0" and not l["erro_execucao"]]
    fp = sum(_bool(l["agente_falha_detectada"]) is True for l in c0)
    print(f"\nFalsos positivos do agente no controle (C0): {fp}/{len(c0)}")

    col_tk = [l.get("tokens_total", "") for l in linhas if not l["erro_execucao"]]
    com_tk = [l for l in linhas if not l["erro_execucao"] and l.get("tokens_total")]
    if col_tk:
        soma = lambda c: sum(int(l[c] or 0) for l in com_tk)
        print(f"\nTokens ({len(com_tk)}/{len(col_tk)} execuções com contagem registrada): "
              f"entrada {soma('tokens_entrada')}, saída {soma('tokens_saida')}, "
              f"raciocínio {soma('tokens_raciocinio')}, total {soma('tokens_total')}")


def main():
    p = argparse.ArgumentParser(description="Experimento do TCC II")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rodar")
    r.add_argument("--seco", action="store_true", help="não chama a API; usa respostas falsas")
    r.add_argument("--retomar", metavar="PASTA", help="continua uma execução interrompida nessa pasta")
    s = sub.add_parser("resumir")
    s.add_argument("arquivo")
    a = p.parse_args()
    if a.cmd == "rodar":
        rodar(a.seco, a.retomar)
    else:
        resumir(a.arquivo)


if __name__ == "__main__":
    sys.exit(main())

# core.py

import json
import time
import re
import os
import logging
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Configurar logging
logging.basicConfig(level=logging.INFO)

class ModeloLLM:
    def __init__(self, nome_modelo="llama3.2"):
        self.modelo = Ollama(model=nome_modelo)
    
    def gerar(self, prompt, max_tokens, temperatura):
        return make_api_call(self.modelo, prompt, max_tokens, temperatura)

def make_api_call(model, prompt, max_tokens, temperature):
    for attempt in range(3):
        try:
            logging.info(f"Tentativa {attempt + 1} de 3 para gerar resposta.")
            response = model.generate([prompt], model_kwargs={"max_tokens": max_tokens, "temperature": temperature})
            generated_text = response.generations[0][0].text.strip()

            logging.info("Resposta gerada pelo modelo:")
            logging.info(generated_text)

            # Simula o streaming de tokens (Ajuste conforme necessário)
            for token in generated_text.split():
                print(token, end=' ', flush=True)
                time.sleep(0.05)
            print()

            return generated_text
        except Exception as e:
            logging.error(f"Exception: {str(e)}")
            if attempt == 2:
                return f"Falha ao gerar resposta após 3 tentativas. Erro: {str(e)}"
            time.sleep(1)

def carregar_passos_padrao():
    arquivo_passos = 'passos_padrao.txt'
    passos_padrao = [
        "Compreensão da pergunta",
        "Identificação dos dados relevantes",
        "Formulação de hipóteses",
        "Análise lógica",
        "Verificação da consistência",
        "Consideração de alternativas",
        "Síntese da resposta",
        "Revisão e refinamento",
        "Verificação da lógica"
    ]
    
    if not os.path.exists(arquivo_passos):
        with open(arquivo_passos, 'w') as file:
            for passo in passos_padrao:
                file.write(f"{passo}\n")
    
    with open(arquivo_passos, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def responde_chain_of_thought(modelo, pergunta, vectorstore):
    lista_passos = carregar_passos_padrao()
    cadeias_aprovadas = get_similar_chains(vectorstore, pergunta)
    cadeias_texto = [doc.page_content for doc in cadeias_aprovadas]
    return processo_raciocinio_completo(modelo, pergunta, cadeias_texto, lista_passos)

def processo_raciocinio_completo(modelo, pergunta, cadeias_aprovadas, lista_passos, max_iteracoes=10):
    cadeia_inicial = gerar_cadeia_raciocinio(modelo, pergunta, cadeias_aprovadas)
    passos_executados = [cadeia_inicial]
    
    for iteracao in range(max_iteracoes):
        avaliacao = avaliar_proximo_passo(modelo, pergunta, "\n".join(passos_executados), lista_passos)
        if "Resposta final" in avaliacao:
            break
        
        proximo_passo = avaliacao.split(": ")[1]
        resultado_passo = executar_passo(modelo, pergunta, proximo_passo, "\n".join(passos_executados))
        passos_executados.append(f"{proximo_passo}:\n{resultado_passo}")
        
        if iteracao == max_iteracoes - 1:
            passos_executados.append("Aviso: Limite máximo de iterações atingido.")
    
    resposta_final = sintetizar_resposta_final(modelo, pergunta, "\n".join(passos_executados))
    return resposta_final, passos_executados

def gerar_cadeia_raciocinio(modelo, pergunta, cadeias_aprovadas):
    cadeias_relevantes = []
    for cadeia in cadeias_aprovadas:
        if avaliar_relevancia_cadeia(modelo, pergunta, cadeia):
            cadeia_preparada = preparar_cadeia_para_prompt(modelo, pergunta, cadeia)
            cadeias_relevantes.append(cadeia_preparada)
    
    cadeias_texto = "\n\n".join(cadeias_relevantes)
    
    prompt = f"""
    Pergunta: {pergunta}

    Com base nas seguintes cadeias de raciocínio aprovadas e relevantes:
    {cadeias_texto}

    Gere uma cadeia de raciocínio inicial para responder à pergunta, seguindo uma sequência lógica de passos.
    """
    return modelo.gerar(prompt, max_tokens=500, temperatura=0.7)

def avaliar_relevancia_cadeia(modelo, pergunta, cadeia):
    prompt = f"""
    Pergunta: {pergunta}

    Cadeia de raciocínio:
    {cadeia}

    Esta cadeia de raciocínio é relevante para responder à pergunta acima?
    Responda apenas com "Sim" ou "Não".
    """
    resposta = modelo.gerar(prompt, max_tokens=10, temperatura=0.3)
    return resposta.strip().lower() == "sim"

def avaliar_proximo_passo(modelo, pergunta, passos_anteriores, lista_passos):
    prompt = f"""
    Pergunta: {pergunta}

    Passos anteriores:
    {passos_anteriores}

    Com base nos passos anteriores, determine se devemos continuar com mais um passo ou se já temos a resposta final.
    Se precisarmos de mais um passo, escolha o próximo passo mais apropriado da seguinte lista:
    {lista_passos}

    Responda apenas com "Próximo passo: [nome do passo]" ou "Resposta final".
    """
    return modelo.gerar(prompt, max_tokens=50, temperatura=0.3)

def executar_passo(modelo, pergunta, passo, passos_anteriores):
    prompt = f"""
    Pergunta: {pergunta}

    Passos anteriores:
    {passos_anteriores}

    Execute o seguinte passo: {passo}
    Forneça uma análise detalhada para este passo.
    """
    return modelo.gerar(prompt, max_tokens=300, temperatura=0.7)

def sintetizar_resposta_final(modelo, pergunta, todos_passos):
    prompt = f"""
    Pergunta original: {pergunta}

    Com base nos seguintes passos de raciocínio:
    {todos_passos}

    Por favor, forneça uma resposta final completa e detalhada para a pergunta.
    """
    return modelo.gerar(prompt, max_tokens=500, temperatura=0.5)

def resumir_cadeia(modelo, cadeia, max_tokens=200):
    prompt = f"""
    Resuma a seguinte cadeia de raciocínio em no máximo {max_tokens} tokens:

    {cadeia}
    """
    return modelo.gerar(prompt, max_tokens=max_tokens, temperatura=0.7)

def preparar_cadeia_para_prompt(modelo, pergunta, cadeia, max_tokens=500):
    if len(cadeia.split()) > max_tokens:
        cadeia_resumida = resumir_cadeia(modelo, cadeia, max_tokens)
        return f"{cadeia_resumida}\n\nLembre-se da pergunta original: {pergunta}"
    return cadeia

def initialize_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    vectorstore_path = "approved_chains"
    if not os.path.exists(vectorstore_path):
        os.makedirs(vectorstore_path)
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    return vectorstore

def get_similar_chains(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    if results:
        return results
    return []

def approve_chain(vectorstore, steps, total_time):
    chain_text = "\n".join([f"{title}\n{content}" for title, content, _ in steps])
    doc = Document(page_content=chain_text, metadata={"total_time": total_time})
    vectorstore.add_documents([doc])
    vectorstore.persist()
    return "Cadeia de raciocínio aprovada e armazenada com sucesso!"
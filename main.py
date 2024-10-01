import gradio as gr
import json
import time
import re
import os
import sys
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def make_api_call(model, prompt, max_tokens):
    for attempt in range(3):
        try:
            response = model.generate([prompt], model_kwargs={"max_tokens": max_tokens})
            generated_text = response.generations[0][0].text.strip()
            
            # Simula o streaming de tokens (Ajuste conforme necessário)
            for token in generated_text.split():
                print(token, end=' ', flush=True)
                time.sleep(0.05)  # Intervalo ajustável para simulação
            print()  # Nova linha após a geração completa

            json_match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                return json.loads(json_text)
            else:
                raise json.JSONDecodeError("JSON não encontrado na resposta.", generated_text, 0)
        except json.JSONDecodeError:
            if attempt == 2:
                return {"title": "Erro", "content": "A resposta não está no formato JSON esperado.", "next_action": "final_answer"}
        except Exception as e:
            if attempt == 2:
                return {
                    "title": "Erro",
                    "content": f"Falha ao gerar resposta após 3 tentativas. Erro: {str(e)}",
                    "next_action": "final_answer"
                }
            time.sleep(1)

def initialize_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    vectorstore_path = "approved_chains"

    # Verifica se o diretório do vectorstore existe; se não, cria-o
    if not os.path.exists(vectorstore_path):
        os.makedirs(vectorstore_path)

    # Inicializa o vectorstore
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    return vectorstore

def get_similar_chains(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    if results:
        return results
    return []

def generate_response(user_prompt, similar_chains):
    model = Ollama(model="llamaPT")

    # Formata as cadeias de raciocínio similares para incluir no prompt
    examples = "\n\n".join([f"Exemplo {i+1}:\n{doc.page_content}" for i, doc in enumerate(similar_chains)])

    system_prompt = f"""Você é um assistente AI que explica seu raciocínio passo a passo em português. Para cada passo, forneça um título e um conteúdo. Decida se precisa continuar ou se está pronto para dar a resposta final. Responda em formato JSON com as chaves "title", "content" e "next_action" (sendo "continue" ou "final_answer"). Use suas habilidades completas, incluindo a geração de código quando necessário. Use o raciocínio chain-of-thought para chegar à resposta. Mantenha o contexto acumulado para referência futura.

Exemplos de cadeias de raciocínio aprovadas anteriormente:

{examples}

Exemplo de cadeia de raciocínio sofisticada:

1. Compreensão da pergunta
2. Identificação dos dados relevantes
3. Formulação de hipóteses
4. Análise lógica
5. Verificação da consistência
6. Consideração de alternativas
7. Síntese da resposta
8. Revisão e refinamento

Inclua etapas de verificação da lógica em seu processo de raciocínio."""

    accumulated_context = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Usuário: "{user_prompt}"'}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        conversation = "\n".join([f"{m['content']}" for m in messages])
        prompt_with_context = f"{conversation}\n\nContexto acumulado:{accumulated_context}\n\nAssistente:"

        start_time = time.time()
        step_data = make_api_call(model, prompt_with_context, 500)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        if not all(key in step_data for key in ('title', 'content', 'next_action')):
            step_data = {"title": "Erro", "content": "Resposta inválida ou incompleta.", "next_action": "final_answer"}

        accumulated_context += f"\nPasso {step_count} - {step_data['title']}:\n{step_data['content']}"
        steps.append((f"Passo {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        messages.append({"role": "assistant", "content": f"```json\n{json.dumps(step_data, ensure_ascii=False)}\n```"})

        yield steps, total_thinking_time

        if step_data['next_action'] == 'final_answer' or step_count >= 8:
            break

        step_count += 1

    final_prompt = f"{conversation}\n\nContexto acumulado:{accumulated_context}\n\nAssistente:"
    start_time = time.time()
    final_data = make_api_call(model, final_prompt, 500)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if not all(key in final_data for key in ('title', 'content')):
        final_data = {"title": "Erro", "content": "Resposta inválida ou incompleta."}

    steps.append(("Resposta Final", final_data.get('content', 'Resposta não disponível.'), thinking_time))
    yield steps, total_thinking_time

def process_query(query):
    output = ""
    steps = []
    total_time = 0
    similar_chains = get_similar_chains(vectorstore, query)
    for step_result, step_time in generate_response(query, similar_chains):
        steps = step_result
        total_time += step_time
        output = ""
        for title, content, time in steps:
            output += f"### {title}\n{content}\n\n"
        output += f"\nTempo total de processamento: {total_time:.2f} segundos"
    return output, steps, total_time

def approve_chain(steps, total_time):
    chain_text = "\n".join([f"{title}\n{content}" for title, content, _ in steps])
    doc = Document(page_content=chain_text, metadata={"total_time": total_time})
    vectorstore.add_documents([doc])
    vectorstore.persist()
    return "Cadeia de raciocínio aprovada e armazenada com sucesso!"

def get_similar_chain_display(query):
    similar_chains = get_similar_chains(vectorstore, query)
    if similar_chains:
        return "\n\n".join([f"**Cadeia {i+1}:**\n{doc.page_content}" for i, doc in enumerate(similar_chains)])
    return "Nenhuma cadeia de raciocínio similar encontrada."

def main():
    global vectorstore
    vectorstore = initialize_vectorstore()

    with gr.Blocks() as iface:
        gr.Markdown("# Assistente AI com Cadeia de Raciocínio")
        gr.Markdown("Este é um protótipo que usa prompting para criar cadeias de raciocínio sofisticadas para melhorar a precisão da saída.")

        with gr.Row():
            input_text = gr.Textbox(lines=2, placeholder="Digite sua pergunta aqui...", label="Pergunta")
            similar_chain_display = gr.Markdown(label="Cadeias de Raciocínio Similares")

        submit_btn = gr.Button("Enviar")
        output = gr.Markdown(label="Resposta")
        
        approve_btn = gr.Button("Aprovar cadeia de raciocínio")
        approve_output = gr.Textbox(label="Status da Aprovação", interactive=False)
        
        steps_state = gr.State([])
        total_time_state = gr.State(0)

        def on_submit(query):
            similar_display = get_similar_chain_display(query)
            response, new_steps, new_time = process_query(query)
            return response, similar_display, new_steps, new_time

        submit_btn.click(
            on_submit,
            inputs=[input_text],
            outputs=[output, similar_chain_display, steps_state, total_time_state]
        )

        approve_btn.click(
            approve_chain,
            inputs=[steps_state, total_time_state],
            outputs=[approve_output]
        )

    iface.launch()

if __name__ == "__main__":
    main()
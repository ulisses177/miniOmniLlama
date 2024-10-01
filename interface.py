# interface.py

import gradio as gr
from core import initialize_vectorstore, process_query, approve_chain
import logging
import re

logging.basicConfig(level=logging.INFO)

def get_similar_chain_display(vectorstore, query):
    similar_chains = vectorstore.similarity_search(query, k=3)
    if similar_chains:
        return "\n\n".join([f"**Cadeia {i+1}:**\n{doc.page_content}" for i, doc in enumerate(similar_chains)])
    return "Nenhuma cadeia de raciocínio similar encontrada."

def main():
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
            logging.info(f"Processando consulta: {query}")
            similar_display = get_similar_chain_display(vectorstore, query)
            generator = process_query(vectorstore, query)
            steps = []
            total_time = 0
            output_text = ""
            for step_output, step_time in generator:
                steps.append((step_output, step_time))
                total_time += step_time
                output_text += f"{step_output}"
                yield output_text, similar_display, steps, total_time
            logging.info("Consulta processada.")

        submit_btn.click(
            on_submit,
            inputs=[input_text],
            outputs=[output, similar_chain_display, steps_state, total_time_state],
            queue=True
        )

        def on_approve(steps, total_time):
            logging.info("Aprovando cadeia de raciocínio.")
            # Extrai o título e conteúdo de cada passo
            chain_steps = []
            for step in steps:
                step_output, step_time = step
                match = re.match(r"### (.*)\n(.*)\n\n", step_output, re.DOTALL)
                if match:
                    title = match.group(1)
                    content_text = match.group(2)
                    chain_steps.append((title, content_text, step_time))
            # Aprova a cadeia de raciocínio
            status = approve_chain(vectorstore, chain_steps, total_time)
            logging.info("Cadeia de raciocínio aprovada.")
            return status

        approve_btn.click(
            on_approve,
            inputs=[steps_state, total_time_state],
            outputs=[approve_output]
        )

    iface.launch()

if __name__ == "__main__":
    main()

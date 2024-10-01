# interface.py

import gradio as gr
from core import initialize_vectorstore, responde_chain_of_thought, approve_chain, ModeloLLM
import logging
import re

logging.basicConfig(level=logging.INFO)

def main():
    vectorstore = initialize_vectorstore()
    modelo = ModeloLLM()

    with gr.Blocks() as iface:
        gr.Markdown("# Assistente AI com Cadeia de Raciocínio")
        gr.Markdown("Este é um protótipo que usa prompting para criar cadeias de raciocínio sofisticadas para melhorar a precisão da saída.")

        input_text = gr.Textbox(lines=2, placeholder="Digite sua pergunta aqui...", label="Pergunta")
        submit_btn = gr.Button("Enviar")
        output = gr.Markdown(label="Passos de Raciocínio")
        final_answer = gr.Markdown(label="Resposta Final")  # Mudamos para Markdown
        
        approve_btn = gr.Button("Aprovar cadeia de raciocínio")
        approve_output = gr.Textbox(label="Status da Aprovação", interactive=False)
        
        steps_state = gr.State([])
        total_time_state = gr.State(0)

        def on_submit(query):
            logging.info(f"Processando consulta: {query}")
            resposta_final, passos = responde_chain_of_thought(modelo, query, vectorstore)
            
            output_text = "\n\n".join([f"### {passo}" for passo in passos])
            resposta_final_markdown = f"## Resposta Final\n\n{resposta_final}"
            
            return output_text, resposta_final_markdown, passos, 0  # 0 para total_time por enquanto

        submit_btn.click(
            on_submit,
            inputs=[input_text],
            outputs=[output, final_answer, steps_state, total_time_state],
            queue=True
        )

        def on_approve(steps, total_time):
            logging.info("Aprovando cadeia de raciocínio.")
            chain_steps = []
            for step in steps:
                match = re.match(r"(.*?):\n(.*)", step, re.DOTALL)
                if match:
                    title = match.group(1)
                    content_text = match.group(2)
                    chain_steps.append((title, content_text, 0))  # 0 para step_time por enquanto
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
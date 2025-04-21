import os
from langchain_openai import ChatOpenAI
from docload import PDFLoader
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "CapituloX"

def main():
    st.set_page_config(page_title="Chat com arquivos pdf", layout="centered")
    st.title("🤖 Pergunte ao CapítuloX – ")
    
    uploaded_files = st.file_uploader("**Upload your pdf Here**", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.info("Envie um Arquivo")
        st.stop()

    similarity_query = st.text_area(
    "🔍 Escreva o que você quer extrair dos arquivos (consulta de similaridade):",
    )

    @st.cache_data
    def load_context_from_pdfs(uploaded_files, query):
        pdf_loader = PDFLoader()
        extracted_context = pdf_loader.art_load_pdf_and_extract_key_content(uploaded_files, query)
        return "\n\n\n".join(extracted_context)

    with st.spinner("Extraindo conteudo dos arquivos..."):
        combined_contexts = load_context_from_pdfs(uploaded_files, similarity_query)
    if not combined_contexts:
        st.stop()
    
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    system_template = (
        "Você é uma IA que responde perguntas com base no seguinte contexto:\n{context}\n"
        "Sempre relacione o conteúdo com a pergunta do usuário."
    )
    sys_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    chat_template = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])
    chain = chat_template | llm
    human_question = st.chat_input("Digite o que você quer perguntar baseado nos arquivos enviados: ")
    if human_question:
        with st.spinner("Gerando resposta..."):
            try:
                response = chain.invoke({
                    "context": combined_contexts,
                    "input": human_question
                })
                st.markdown("### 💬 Resposta")
                st.write(response.content)
            except Exception as e:
                st.error(f"Ocorreu um erro ao gerar a resposta: {e}")

    with st.expander("📚 Ver conteúdo extraído dos PDFs"):
        st.write(combined_contexts)

if __name__ == "__main__":
    main()

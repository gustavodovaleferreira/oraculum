import streamlit as st
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from utils import get_by_session_id
from faiss_db import search_documents
from dotenv import load_dotenv
import os
import re  # Usado para remover [Fonte X] e similares

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")


def clear_session_id():
    """Limpa o ID da sessão e reinicia o histórico"""
    st.session_state.session_id_chat = None
    if "session_id_chat" in st.session_state:
        get_by_session_id(st.session_state.session_id_chat).clear()


def load_llm():
    """Configura o pipeline de LLM com suporte a RAG especializado em estética"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente altamente especializado nas áreas de estética facial, estética corporal e procedimentos estéticos avançados. 
Baseie suas respostas exclusivamente nas informações fornecidas no contexto abaixo:

{context}

Caso o contexto não contenha informações relevantes à pergunta, informe isso de forma clara e respeitosa ao usuário. 
Nunca forneça informações que não estejam no conteúdo fornecido, e jamais invente ou suponha dados.

Utilize o conteúdo com precisão, e exiba os nomes dos arquivos ou referências de fonte nas respostas.

Referencie o nome do arquivo .md conforme a fonte utilizada.

Adote um tom profissional e objetivo. Explique termos técnicos de forma acessível, sempre que necessário, mantendo clareza e precisão nas respostas."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return prompt | ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0.5,
        model=MODEL_CHAT,
        streaming=True
    )


def show():
    st.title("Interface de Chat com RAG")
    st.write("Área para interação via chat utilizando RAG para buscar informações na base FAISS.")

    with st.sidebar:
        st.header("Opções de Chat")
        st.button("Limpar Sessão", on_click=clear_session_id)

    if not st.session_state.get("session_id_chat"):
        st.session_state.session_id_chat = str(uuid4())

    chain = load_llm()
    history = get_by_session_id(st.session_state.session_id_chat)

    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)

    if prompt := st.chat_input("Digite sua mensagem"):
        human_message = HumanMessage(content=prompt)
        history.add_messages([human_message])
        st.chat_message("human").markdown(prompt)

        try:
            docs = search_documents(prompt, k=10)
            context = ""
            for i, (doc, score) in enumerate(docs):
                source = doc.metadata.get('source', 'Fonte desconhecida')
                context += f"Fonte {i + 1} ({source}): {doc.page_content}\n\n"
        except Exception as e:
            st.error(f"Erro na busca de contexto: {str(e)}")
            context = "Nenhum contexto encontrado."
            
        chat_history = history.messages[:-1]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({
                    "question": prompt,
                    "history": chat_history,
                    "context": context
                }):
                    if content := getattr(chunk, 'content', ''):
                        full_response += content
                        # Limpeza parcial durante o stream
                        partial_clean = re.sub(r"\[\s*[^]]+\s*\]", "", full_response)
                        response_placeholder.markdown(partial_clean + "▌")

                # Limpeza final da resposta antes de exibir e salvar
                cleaned_response = re.sub(r"\[\s*[^]]+\s*\]", "", full_response)
                response_placeholder.markdown(cleaned_response.strip())
                history.add_messages([AIMessage(content=cleaned_response.strip())])

            except Exception as e:
                st.error(f"Erro na geração da resposta: {str(e)}")
                history.add_messages([AIMessage(content="Desculpe, ocorreu um erro interno.")])
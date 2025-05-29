import streamlit as st
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from utils import get_by_session_id
from faiss_db import search_documents
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def clear_session_id():
    st.session_state.session_id_chat = None
    if "session_id_chat" in st.session_state:
        get_by_session_id(st.session_state.session_id_chat).clear()

def load_llm():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Você é um assistente especializado em estética facial, corporal e procedimentos estéticos.

Baseie suas respostas exclusivamente no conteúdo fornecido abaixo como contexto, extraído de documentos técnicos:

{context}

Regras obrigatórias:
1. Para cada informação que você extrair do contexto acima, cite logo após a frase, no formato: [Fonte: nome-do-arquivo.ext].
   Exemplo: "A acne é uma condição inflamatória da pele [Fonte: acne.md]."

2. Para toda informação que **não constar no contexto acima**, você deve marcar a frase com: [Sem fonte].
   Exemplo: "Essa condição pode afetar a autoestima [Sem fonte]."

3. NÃO RESUMA. NÃO AGRUPE fontes. CITE após cada afirmação.

4. Mantenha tom técnico, claro e profissional. Explique termos se necessário.

IMPORTANTE: Toda frase precisa indicar a origem: [Fonte: ...] ou [Sem fonte]. Isso é obrigatório.
"""),
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
    st.write("Chat especializado com referências extraídas diretamente da base FAISS.")

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
            for doc, score in docs:
                source = doc.metadata.get('source', 'Fonte desconhecida')
                context += f"{doc.page_content}\n[Fonte: {source}]\n\n"
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
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response.strip())
                history.add_messages([AIMessage(content=full_response.strip())])

                if "[Fonte:" not in full_response and "[Sem fonte]" not in full_response:
                    st.warning("⚠️ A resposta não indicou nenhuma fonte. Pode ter ignorado o contexto.")

            except Exception as e:
                st.error(f"Erro na geração da resposta: {str(e)}")
                history.add_messages([AIMessage(content="Desculpe, ocorreu um erro interno.")])

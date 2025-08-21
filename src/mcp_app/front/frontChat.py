import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime import get_instance
from src.mcp_app.back.chat.backChat import query_llm

def obtener_id_sesion():
    # Obtener la instancia de runtime de Streamlit
    runtime = get_instance()
    # Obtener el contexto de ejecución del script
    ctx = get_script_run_ctx()
    if ctx:
        # Obtener el ID de la sesión actual
        session_id = ctx.session_id
        return session_id
    else:
        return None

def render_page():
    # Añadir el botón "Volver Atrás"
    if st.button("Volver al Menú Principal"):
        st.session_state.page = None
        st.rerun()
    st.header("Página de Chat")

    #st.title("Chatbot con RAG y Recuperación de Grafos")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Entrada del usuario
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    if user_input:
        st.session_state['messages'].append({'type': 'user', 'content': user_input})

    # Mostrar mensajes
    for message in st.session_state['messages']:
        if message['type'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])

    # Obtener respuesta del backend solo si hay entrada del usuario
    if user_input:

        response_text = query_llm(user_input,obtener_id_sesion())
        bot_response = f"{response_text}"
        st.session_state['messages'].append({'type': 'assistant', 'content': bot_response})

        # Mostrar la respuesta
        with st.chat_message("assistant"):
            st.write(bot_response)
if __name__ == "__main__":
    render_page()
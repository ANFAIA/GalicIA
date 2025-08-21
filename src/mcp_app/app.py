import streamlit as st
from src.mcp_app.front.frontChat import render_page as render_chat_page
from src.mcp_app.services.init.vLLM import init as initVLLM
from src.mcp_app.back.chat.mcp_server import ensure_fastmcp_running
from src.mcp_app.services.init.download_models import innit_model


def main():
    st.title("galicIA")
    # Inicialización única usando st.session_state
    if 'initialized' not in st.session_state:
        ensure_fastmcp_running()
        #initVLLM()
        innit_model()
        st.session_state.initialized = True
        st.success("Inicialización completada.")

    # Inicialización única usando st.session_state
    if 'initialized' not in st.session_state:
        ensure_fastmcp_running()
        st.session_state.initialized = True
        st.success("Inicialización completada.")

    # Configuración de la página en session_state
    if 'page' not in st.session_state:
        st.session_state.page = None

    # Renderizado de páginas según el estado
    if st.session_state.page == 'chat':
        render_chat_page()

    else:
        st.write("Seleccione una opción:")
        col1 = st.columns(1)
        with col1:
            if st.button("Abrir Chat"):
                st.session_state.page = 'chat'
                st.rerun()


if __name__ == "__main__":
    main()


import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Divide el texto en fragmentos y los convierte en formato de documento
def chunks_and_document(txt):
    text_splitter = CharacterTextSplitter()  # Divide el texto por caracteres
    texts = text_splitter.split_text(txt)  # Divide el texto en fragmentos más pequeños
    docs = [Document(page_content=t) for t in texts]  # Convierte los fragmentos en formato de documento
    return docs
    
# Carga del modelo Llama 2
def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])  # Instancia del callback con un manejador de salida de texto en tiempo real
    llm = CTransformers(
        model='C:/Users/SESA716043/Desktop/TAAD - Actividad 2/llama-2-7b-chat.ggmlv3.q2_K.bin',  # Reemplaza con la ruta local al modelo Llama 2
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm
 
# Aplica el modelo LLM a nuestro documento
def chains_and_response(docs):
    llm = load_llm()
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)
    
# Configuración de la página
st.set_page_config(page_title='🦜🔗 Aplicación de Resumen de Texto')
st.title('🦜🔗 Aplicación de Resumen de Texto')

# Entrada de texto
txt_input = st.text_area('Introduce tu texto', '', height=200)

# Formulario para aceptar la entrada de texto del usuario para el resumen
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Enviar')
    if submitted:
        with st.spinner('Calculando...'):
            docs = chunks_and_document(txt_input)
            response = chains_and_response(docs)
            result.append(response)

# Muestra el resultado del resumen
if len(result):
    st.title('📝✅ Resultado del Resumen')
    st.info(response)

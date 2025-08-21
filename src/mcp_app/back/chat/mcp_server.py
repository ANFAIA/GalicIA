from mcp.server.fastmcp import FastMCP
import socket
import threading
import time
from src.mcp_app.services.tools.tools import generar_poema as generar_poema_service
import configparser

config = configparser.ConfigParser()

mcp = FastMCP("poema_mcp")

@mcp.tool()
async def generar_poema(promt: str)->str:
    """
    Dado un promt genera un poema
    """
    return generar_poema_service(promt)


def ensure_fastmcp_running(host="localhost", port=8000, timeout=2):
    """
    Verifica si el servidor FastMCP está corriendo (por ejemplo,
    comprobando si el puerto 'port' en 'host' responde) y, si no,
    lo inicia en un hilo en segundo plano.

    Args:
        host (str): El host donde se espera que esté levantado el servidor.
        port (int): El puerto donde se espera que escuche.
        timeout (int): Tiempo máximo (en segundos) para la comprobación.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        print("FastMCP ya se está ejecutando en {}:{}".format(host, port))
    except socket.error:
        print("FastMCP no se encontró en {}:{}; se va a iniciar...".format(host, port))
        def run_server():
            try:
                # Este método es bloqueante, por lo que se ejecuta en un hilo daemon.
                mcp.run(transport="sse")
            except Exception as e:
                print("Error al iniciar FastMCP:", e)
        threading.Thread(target=run_server, daemon=True).start()
        # Dar tiempo para que el servidor se inicie
        time.sleep(3)
        print("FastMCP debería estar corriendo ahora en {}:{}".format(host, port))

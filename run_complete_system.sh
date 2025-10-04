#!/bin/bash

# Seismic AI Complete System Launcher
# Inicia el servidor API y ejecuta la demo del cliente

echo "🌟 Seismic AI Complete System Launcher"
echo "======================================"
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para verificar si un puerto está en uso
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Puerto $1 ya está en uso${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Puerto $1 disponible${NC}"
        return 0
    fi
}

# Función para esperar a que el servidor esté listo
wait_for_server() {
    local port=$1
    local timeout=30
    local count=0

    echo -e "${BLUE}⏳ Esperando a que el servidor esté listo en puerto $port...${NC}"

    while ! curl -s http://127.0.0.1:$port/health > /dev/null; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout esperando al servidor${NC}"
            return 1
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo -e "\n${GREEN}✅ Servidor listo!${NC}"
    return 0
}

# Verificar puerto 8000
if ! check_port 8000; then
    echo -e "${YELLOW}💡 El servidor podría estar ya ejecutándose${NC}"
    echo -e "${YELLOW}   Intentando conectar al servidor existente...${NC}"

    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo -e "${GREEN}✅ Servidor ya ejecutándose, procediendo con la demo${NC}"
        SERVER_RUNNING=true
    else
        echo -e "${RED}❌ No se puede conectar al servidor existente${NC}"
        echo -e "${YELLOW}💡 Inicia el servidor manualmente: python backend/simple_api_server.py${NC}"
        exit 1
    fi
else
    SERVER_RUNNING=false
fi

# Iniciar servidor si no está ejecutándose
if [ "$SERVER_RUNNING" = false ]; then
    echo -e "${BLUE}🚀 Iniciando servidor API...${NC}"

    # Iniciar servidor en background
    python ../backend/simple_api_server.py &
    SERVER_PID=$!

    # Esperar a que el servidor esté listo
    if ! wait_for_server 8000; then
        echo -e "${RED}❌ Error iniciando el servidor${NC}"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi

    echo -e "${GREEN}✅ Servidor API iniciado (PID: $SERVER_PID)${NC}"
fi

echo ""
echo -e "${BLUE}🎯 Ejecutando demo del cliente API...${NC}"
echo ""

# Ejecutar la demo del cliente
python ../backend/api_client_demo.py

echo ""
echo -e "${GREEN}🎉 Demo completada!${NC}"
echo ""
echo -e "${BLUE}📚 Recursos disponibles:${NC}"
echo -e "   📖 Documentación API: ${YELLOW}http://127.0.0.1:8000/docs${NC}"
echo -e "   🔄 ReDoc: ${YELLOW}http://127.0.0.1:8000/redoc${NC}"
echo -e "   🏥 Health Check: ${YELLOW}http://127.0.0.1:8000/health${NC}"

if [ "$SERVER_RUNNING" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  El servidor sigue ejecutándose en background${NC}"
    echo -e "${YELLOW}   Para detenerlo: ${NC}kill $SERVER_PID"
    echo -e "${YELLOW}   O presiona Ctrl+C${NC}"
fi

echo ""
echo -e "${GREEN}¡Gracias por usar Seismic AI! 🌟${NC}"
import sys
from search import search_prompt
from langchain_openai import ChatOpenAI

def main():
    model = ChatOpenAI(model="gpt-5-mini", temperature=0) # type: ignore
    chain = search_prompt | model

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    while True:
        try: 
            user_input = input("Digite sua pergunta (ou 'sair' para encerrar): ")
            if user_input.lower() in {"sair"}:
                print("\nEncerrando o chat. Até mais!")
                break

            if not user_input.strip():
                continue
            
            print("Pesquisando e gerando resposta...", end="", flush=True)

            response = chain.invoke(user_input)
            print(f"\nRESPOSTA: {response.content}\n")

        except Exception as e:
            print(f"\nOcorreu um erro ao processar sua pergunta: {e}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat encerrado abruptamente. Até mais!")
        sys.exit(0)
    except Exception as e:
        print(f"Ocorreu um erro fatal inesperado: {e}")
        sys.exit(1)
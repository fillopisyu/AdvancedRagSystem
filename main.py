from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    while True:
        try:
            question = input("Sorunuzu giriniz (Çıkmak için 'q' ya basın): ")

            if question.lower() == 'q':
                print("Program sonlandırılıyor...")
                break

            result = app.invoke(input={"question": question})
            print("\nCevap:", result.get('generation', ''))
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nProgram sonlandırılıyor...")
            break
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
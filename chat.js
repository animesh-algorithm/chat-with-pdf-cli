import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { BufferMemory } from "langchain/memory";
import readline from "readline";

async function askQuestion(chain, question) {
  const res = await chain.call({ question });
  console.log(res.text);
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askDocumentPath = async () => {
    rl.question(
      "\x1b[1mEnter the path to the document: \x1b[0m",
      async (documentPath) => {
        const loader = new PDFLoader(documentPath);

        let docs = await loader.load();
        const model = new OpenAI({});
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 200,
        });

        docs = await splitter.splitDocuments(docs);

        const vectorStore = await HNSWLib.fromDocuments(
          docs,
          new OpenAIEmbeddings()
        );

        const chain = ConversationalRetrievalQAChain.fromLLM(
          model,
          vectorStore.asRetriever(),
          {
            memory: new BufferMemory({
              memoryKey: "chat_history",
            }),
          }
        );

        const askInitialQuestion = async () => {
          rl.question(
            "\x1b[1mEnter your question:\x1b[0m ",
            async (question) => {
              await askQuestion(chain, question);
              handleFollowUpQuestion(chain);
            }
          );
        };

        const handleFollowUpQuestion = async () => {
          rl.question(
            "\x1b[1mEnter your question (or type 'exit' to quit):\x1b[0m ",
            async (followUpQuestion) => {
              if (followUpQuestion.toLowerCase() === "exit") {
                rl.close();
                process.exit(0);
                return;
              }

              await askQuestion(chain, followUpQuestion);
              handleFollowUpQuestion();
            }
          );
        };

        await askInitialQuestion();
      }
    );
  };

  await askDocumentPath();
}

main().catch((error) => {
  console.error(error);
});

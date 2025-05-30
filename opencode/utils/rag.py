from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ChoromaDBManager:
    """管理ChromaDB的连接和工具存储"""
    def __init__(self, db_path: str = "./tool_db", collection_name: str = "tools"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = self.init_chroma_db(db_path, collection_name)
    
    def init_chroma_db(self,db_path, collection_name):
        """初始化ChromaDB连接和集合"""
        client = chromadb.PersistentClient(path=db_path)
        # 包装成 Chroma 可用的嵌入函数
        embedding_func = SentenceTransformerEmbeddingFunction(model_name="/home/zyx/A-Projects/Graduation/embedding/bge-m3", 
                                                        trust_remote_code=True)
        
        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func
        )

    def parse_tool_blocks(self,content: str) -> list[dict]:
        """将Markdown内容按分隔符分割成工具块"""
        blocks = []
        raw_blocks = [b.strip() for b in re.split(r'-{5,}', content) if b.strip()]
        
        for block in raw_blocks:
            lines = block.split('\n')
            tool_name = lines[0].lstrip('#').strip() if lines else "Unnamed Tool"
            blocks.append({
                "name": tool_name,
                "content": '\n'.join(lines[1:]).strip()
            })
        return blocks
    def store_tools_to_db(self, dir_path: str):
        """将目录下的所有.md文件存入数据库"""
        for filename in os.listdir(dir_path):
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(dir_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                tools = self.parse_tool_blocks(f.read())
                
                self.collection.add(
                    documents=[t["content"] for t in tools],
                    metadatas=[{
                        "source_file": filename,
                        "tool_name": t["name"],
                        "block_id": i
                    } for i, t in enumerate(tools)],
                    ids=[f"{filename}_{i}" for i in range(len(tools))]
                )
            print(f"✅ 已存储: {filename} (包含 {len(tools)} 个工具块)")
    def query_tools(self, query: str, n_results: int = 3) -> list[dict]:
        """语义搜索工具文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return [{
            "tool_name": meta["tool_name"],
            "content": doc,
            "source_file": meta["source_file"],
            "distance": dist
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]
    def get_all_tools(self,steps) -> list[dict]:
        """获取所有存储的工具"""
        select_function =[]
        for step in steps:
            # 提取函数名
            query = step.replace("任务: "," ").strip()
            results = self.query_tools(query)
            select_function+=results[0:1]
        return select_function
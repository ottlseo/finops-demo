import boto3
from botocore.config import Config
import json
from typing import List, Dict, Any, Union

class BedrockClient:
    def __init__(self, region: str, llm_model: str):
        self.region = region
        self.llm_model = llm_model
        self.client = self.init_boto3_client(region)
    
    def init_boto3_client(self, region: str):
        """Initialize boto3 client with retry configuration"""
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)
    
    def converse_with_bedrock(self, sys_prompt, usr_prompt):
        """Send a conversation request to Amazon Bedrock"""
        temperature = 0.0
        top_p = 0.1
        top_k = 1
        inference_config = {"temperature": temperature, "topP": top_p}
        additional_model_fields = {"top_k": top_k} if self.llm_model != "anthropic.claude-3-sonnet-20240229-v1:0" else {}
        
        response = self.client.converse(
            modelId=self.llm_model, 
            messages=usr_prompt, 
            system=sys_prompt,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
        return response['output']['message']['content'][0]['text']
    
    def rerank(self, query: str, page_contents: List[Dict[str, Any]]):
        """Rerank search results using Bedrock reranking model"""
        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=self.region)
        model_package_arn = f"arn:aws:bedrock:{self.region}::foundation-model/amazon.rerank-v1:0"
        
        # 빈 page_contents 처리: 빈 경우 기본 응답 반환
        if not page_contents:
            return {"results": []}
        
        text_sources = [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {
                        "text": content['description'], 
                    }
                }
            } for content in page_contents
        ]

        response = bedrock_agent_runtime.rerank(
            queries=[
                {
                    "type": "TEXT",
                    "textQuery": {
                        "text": query
                    }
                }
            ],
            sources=text_sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": 5,
                    "modelConfiguration": {
                        "modelArn": model_package_arn,
                    }
                }
            }
        )
        return response

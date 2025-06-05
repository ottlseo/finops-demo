import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import List, Dict, Any
from langfuse.decorators import observe, langfuse_context

class BedrockClient:
    def __init__(self, region: str, llm_model: str, langfuse_option: bool):
        self.region = region
        self.llm_model = llm_model
        self.client = self.init_boto3_client(region)
        self.langfuse_option = langfuse_option
    
    def init_boto3_client(self, region: str):
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", region_name=region, config=retry_config)
    
    def invoke(self, sys_prompt, usr_prompt, node_name=None):
        """
        langfuse_option 값에 따라 invoke_model 또는 invoke_model_and_log를 호출하는 함수
        """
        if self.langfuse_option:
            return self.invoke_model_with_logging(sys_prompt, usr_prompt, node_name)
        else:
            return self.invoke_model(sys_prompt, usr_prompt)
     
    def invoke_model(self, sys_prompt, usr_prompt):
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
   
    def invoke_model_with_logging(self, sys_prompt, usr_prompt, node_name=None):
        observation_name = node_name if node_name else "Bedrock Converse"
        
        # Dynamically apply the decorator with the specified name
        decorated_func = observe(as_type="generation", name=observation_name)(self.wrapped_bedrock_converse)
        return decorated_func(sys_prompt, usr_prompt)

    @observe(as_type="generation", name="Bedrock Converse")
    def wrapped_bedrock_converse(self, sys_prompt, usr_prompt):
        input_data = {
            "kwargs": {
                "system": sys_prompt,            
                "messages": usr_prompt,
            }
        }    
        inference_config = {"temperature": 0.0, "topP": 0.1}
        # additional_model_fields = {"top_k": 1} if self.llm_model != "anthropic.claude-3-sonnet-20240229-v1:0" else {}
        
        # Langfuse 관측 컨텍스트에 입력, 모델 ID, 파라미터, 기타 메타데이터를 업데이트합니다.
        langfuse_context.update_current_observation(
            input=input_data,
            model=self.llm_model,
            model_parameters=inference_config
        )
        try:
            response = self.client.converse(
                modelId=self.llm_model, 
                messages=usr_prompt, 
                system=sys_prompt,
                inferenceConfig=inference_config
            )
        except (ClientError, Exception) as e:
            error_message = f"ERROR: Can't invoke '{self.llm_model}'. Reason: {e}"
            langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
            print(error_message)
            return
        
        # Langfuse에 출력 텍스트, 토큰 사용량, 응답 메타데이터를 기록합니다.
        response_text = response["output"]["message"]["content"][0]["text"]
        langfuse_context.update_current_observation(
            output=response_text,
            usage_details={
                "input": response["usage"]["inputTokens"],
                "output": response["usage"]["outputTokens"],
                "total": response["usage"]["totalTokens"]
            },
            metadata={
                "ResponseMetadata": response["ResponseMetadata"],
            }
        )
        return response_text
   

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

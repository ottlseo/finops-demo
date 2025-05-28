import boto3
from botocore.config import Config
import json
from typing import List, Dict, Any, Union
from langfuse.decorators import observe, langfuse_context

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
    

    def wrapped_bedrock_converse(self, sys_prompt, usr_prompt):  #**kwargs):

        function_name = "Bedrock Converse"
        # Langfuse 컨텍스트에 이름 업데이트
        langfuse_context.update_current_observation(name=function_name)    

        # 1. extract model metadata
        # kwargs_clone = kwargs.copy()
        system_prompts = sys_prompt #kwargs_clone.pop('system_prompts', None)
        messages = usr_prompt #kwargs_clone.pop('messages', None)
        tool_config = None #kwargs_clone.pop('tool_config', None)
        # system = kwargs_clone.pop('system', None)
        
        # messages와 system을 형식에 맞게 처리
        input_data = {
            "kwargs": {
                "system": system_prompts,            
                "messages": messages,
            }
        }    

        modelId = "anthropic.claude-3-sonnet-20240229-v1:0" #kwargs_clone.pop('modelId', None)

        # model_parameters = {
        #     **kwargs_clone.pop('inferenceConfig', {}),
        # }
        temperature = 0.0
        top_p = 0.1
        top_k = 1
        inference_config = {"temperature": temperature, "topP": top_p}
        # additional_model_fields = {"top_k": top_k} if self.llm_model != "anthropic.claude-3-sonnet-20240229-v1:0" else {}
        
        ##############################3
        # 1. Langfuse 관측 컨텍스트에 입력, 모델 ID, 파라미터, 기타 메타데이터를 업데이트합니다.
        ##############################3
        langfuse_context.update_current_observation(
            input=input_data,
            model=modelId,
            model_parameters=inference_config,
            # metadata=kwargs_clone
        )
        ##############################3
        # 2. model call with error handling
        ##############################3
        import time
        
        max_attempts = 3
        delay_seconds = 1
        response = None
        ai_message = None
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response, ai_message = self.chain(
                    llm=self.llm,
                    system_prompts=system_prompts,
                    messages=messages,
                    tool_config=tool_config,
                    verbose=self.verbose
                )
                # 성공하면 반복문 종료
                break
                
            except (ClientError, Exception) as e:
                last_error = e
                error_message = f"Attempt {attempt + 1}/{max_attempts}: ERROR: Can't invoke '{modelId}'. Reason: {e}"
                print(error_message)
                
                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)
                    continue
                
                # 마지막 시도에서 실패한 경우
                final_error_message = f"ERROR: Can't invoke '{modelId}' after {max_attempts} attempts. Last error: {last_error}"
                langfuse_context.update_current_observation(level="ERROR", status_message=final_error_message)
                print("Error in exception during calling chain: ", final_error_message)
                return {"text": final_error_message}, {"error": final_error_message}
    

        ##############################3
        # 3. extract response metadata
        ##############################3
        # Langfuse에 출력 텍스트, 토큰 사용량, 응답 메타데이터를 기록합니다.
        try:
            response_text = response["text"]
            reasoning_text = response["reasoning"]
            print("## response in wrapped_bedrock_converse: \n", json.dumps(response, indent=2, ensure_ascii=False))

            token_usage = response["token_usage"]
            input_tokens = token_usage.get("inputTokens", 0)
            output_tokens = token_usage.get("outputTokens", 0)
            total_tokens = token_usage.get("totalTokens", 0)

            langfuse_context.update_current_observation(
                output=response_text,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                },
                metadata={
                    "reasoning_text": reasoning_text, 
                    "toolUse_text": response.get("toolUse", {}),
                    # "ResponseMetadata": response.get("ResponseMetadata", {}),
                }
            )

        except (ClientError, Exception) as e:
            print("## response in except in adding to langfuse: \n", response) 
            error_message = f"ERROR: Can't parse:  Reason: {e}"
            langfuse_context.update_current_observation(level="ERROR", status_message=error_message)
            print(error_message)
            # Always return a dict with 'text' key to avoid KeyError downstream
            return {"text": error_message}, {"error": error_message}

        # return {"text": response_text}, ai_message
        return response, ai_message

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

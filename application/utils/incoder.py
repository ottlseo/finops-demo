import boto3
from botocore.exceptions import ClientError
from typing import Optional, Tuple

DYNAMODB_TABLE_NAME = "FinOps-ServiceName"
REGION = "us-east-1"

class ServiceNameNormalizer:
    def __init__(self, table_name: str = DYNAMODB_TABLE_NAME, region: str = REGION):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.service_patterns = self._load_service_patterns()
        
    def _load_service_patterns(self) -> dict: # DDB에서 서비스명 패턴 로드
        try:
            response = self.table.scan( 
                ProjectionExpression='PK, variants, SK'
            )
            patterns = {}
            for item in response.get('Items', []):
                service_code = item.get('SK')
                patterns[item['PK']] = service_code

                for variant in item.get('variants', []):
                    patterns[variant] = service_code
                    
            return patterns
            
        except Exception as e:
            print(f"Error loading service patterns: {e}")
            return {}
    
    def normalize_service_name(self, text: str) -> str:
        return text.lower().strip().replace(" ", "")
    
    def find_service_match(self, text: str) -> Optional[tuple]: # 정규화된 텍스트에서 가장 긴 매칭되는 서비스명 찾기 
        normalized_text = self.normalize_service_name(text)
        
        # 가장 긴 매칭을 찾기 위해 서비스명을 길이순으로 정렬
        matches = []
        for pattern in self.service_patterns.keys():
            if pattern in normalized_text:
                matches.append((pattern, self.service_patterns[pattern]))
        
        # 가장 긴 매칭 반환
        if matches:
            return max(matches, key=lambda x: len(x[0]))
        return None
 
    def process_text(self, text: str) -> Tuple[str, Optional[str]]:
        match = self.find_service_match(text)
        if not match:
            return text, None
        
        service_name, service_code = match
        normalized_input = self.normalize_service_name(text)
        normalized_pattern = self.normalize_service_name(service_name)
        
        # 원본 텍스트에서 매칭되는 부분 찾기
        start_idx = normalized_input.find(normalized_pattern)
        if start_idx != -1:
            # 원본 텍스트에서 해당 위치의 실제 문자열 찾기
            end_idx = start_idx + len(normalized_pattern)
            original_chars = 0
            current_chars = 0
            
            for i, char in enumerate(text):
                normalized_char = self.normalize_service_name(char)
                if not normalized_char:  # 공백이나 특수문자인 경우
                    continue
                    
                if current_chars == start_idx:
                    start_pos = i
                if current_chars == end_idx:
                    end_pos = i
                    break
                current_chars += len(normalized_char)
            else:
                end_pos = len(text)
            
            had_space = end_pos < len(text) and text[end_pos].isspace() # 원본 텍스트에서 서비스명을 서비스 코드로 교체
            
            new_text = text[:start_pos] + service_code
            # 원본에 공백이 있었거나, 다음 문자가 있는 경우에만 공백 추가
            if had_space or end_pos < len(text):
                new_text += ' '
            new_text += text[end_pos:].lstrip()  # 중복 공백 방지를 위해 lstrip() 사용
            
            return new_text, service_code
            
        return text, None
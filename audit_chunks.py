import json
import re
import os

def run_audit(file_path):
    if not os.path.exists(file_path):
        print(f"에러: {file_path} 파일을 찾을 수 없습니다.")
        return

    print(f"[{file_path}] 전수조사 시작...\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    report = {
        "total": len(chunks),
        "llm_noise": [],
        "broken_table": [],
        "empty_summary": [],
        "bad_id": []
    }

    # 혼잣말 필터링 키워드
    filler_keywords = ["요약", "다음과 같", "문서", "제공된", "제공해", "핵심은", "주요 내용", "다음은", "살펴보면"]

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        content = chunk.get("content", "")
        meta = chunk.get("metadata", {})
        summary = meta.get("section_summary", "")
        
        # 1. LLM 혼잣말 검출
        suspected = [kw for kw in filler_keywords if kw in summary]
        if suspected:
            report["llm_noise"].append({"id": cid, "noise": suspected, "text": summary[:40]})
            
        # 2. HTML 표 절단 검출 (<table ...> 과 </table> 개수 불일치 확인)
        if "_table" in cid or "<table" in content:
            if content.count("<table") != content.count("</table") or content.count("<tr") != content.count("</tr"):
                report["broken_table"].append(cid)
                
        # 3. 요약 누락/실패 검출
        if not summary or "실패" in summary or len(summary) < 10:
            report["empty_summary"].append(cid)
            
        # 4. ID 포맷 검출 (예: 00126380_2023_00001_table)
        if not re.match(r'^\d{8}_\d{4}_\d{5}(_table)?$', cid):
            report["bad_id"].append(cid)

    # 결과 출력
    print(f"✅ 총 검사 완료 청크: {report['total']}개")
    print("-" * 50)
    print(f"⚠️ LLM 혼잣말 의심: {len(report['llm_noise'])}건")
    print(f"❌ HTML 표 절단 오류: {len(report['broken_table'])}건")
    print(f"❌ 요약 누락/실패 오류: {len(report['empty_summary'])}건")
    print(f"❌ ID 및 포맷 오류: {len(report['bad_id'])}건")
    print("-" * 50)

    if report["llm_noise"]:
        print(f"\n[혼잣말 샘플 1개 확인]: {report['llm_noise'][0]}")

if __name__ == "__main__":
    run_audit("multi_hybrid_chunks_final.json")
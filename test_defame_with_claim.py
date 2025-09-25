#!/usr/bin/env python3
"""
Test DEFAME with a real claim to see the complete             # Check for social media aggregated evidence  
            social_media_evidence = [e for e in all_evidence if e.action and hasattr(e.action, 'name') and 'social' in e.action.name.lower()]
            if social_media_evidence:
                print(f"   🔗 Social Media Evidence Found: {len(social_media_evidence)} items")
                for j, evidence in enumerate(social_media_evidence, 1):
                    action_name = evidence.action.name if evidence.action else "Unknown"
                    print(f"      {j}. Action: {action_name}")
                    if hasattr(evidence, 'takeaways') and evidence.takeaways:
                        preview = str(evidence.takeaways)[:150] + "..." if len(str(evidence.takeaways)) > 150 else str(evidence.takeaways)
                        print(f"         Content: {preview}")
            
            other_evidence = [e for e in all_evidence if not (e.action and hasattr(e.action, 'name') and 'social' in e.action.name.lower())]ng pipeline in action
"""
import sys
import asyncio
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add scrapeMM to path for Reddit integration
scrapemm_path = current_dir.parent / "scrapeMM"
sys.path.insert(0, str(scrapemm_path))

from defame.fact_checker import FactChecker
from defame.common import Content, Claim
from defame.common.modeling import make_model

def test_fact_check_with_claim():
    """Test DEFAME fact-checking with a real claim"""
    print("🚀 DEFAME Fact-Checking Test")
    print("=" * 60)
    
    # Test claims - you can modify these
    test_claims = [
        {
            "text": "23-year-old Iryna Zarutska is dead",
            "description": "Iryna Zarutska"
        }
    ]
    
    # Initialize the fact checker
    print("🔧 Initializing DEFAME Fact Checker...")
    try:
        # Use GPT-4o-mini for testing (cheaper and faster)
        llm = make_model("gpt_4o_mini")
        fact_checker = FactChecker(llm=llm)
        print("✅ FactChecker initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize FactChecker: {e}")
        return
    
    # Test each claim
    for i, test_case in enumerate(test_claims, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['description']}")
        print(f"📝 Claim: {test_case['text']}")
        print("-" * 60)
        
        try:
            print("🔍 Starting fact-checking process...")
            
            # Use the verify_claim method directly with text
            report, _ = fact_checker.verify_claim(test_case['text'])
            
            print("\n📊 FACT-CHECK RESULTS:")
            print(f"   🎯 Final Verdict: {report.verdict}")
            print(f"   🔢 Confidence: {report.confidence:.2f}" if hasattr(report, 'confidence') else "   🔢 Confidence: Not available")
            
            # Print report summary or content
            if hasattr(report, 'summary') and report.summary:
                summary = report.summary[:200] + "..." if len(report.summary) > 200 else report.summary
                print(f"   📄 Summary: {summary}")
            elif hasattr(report, 'content') and report.content:
                content = str(report.content)[:200] + "..." if len(str(report.content)) > 200 else str(report.content)
                print(f"   📄 Content: {content}")
            
            # Print report structure for debugging
            print(f"   🔍 Report type: {type(report)}")
            print(f"   📋 Report attributes: {[attr for attr in dir(report) if not attr.startswith('_')][:10]}")
            
            # Try to access evidence or other relevant info
            all_evidence = report.get_all_evidence()
            print(f"   📚 Total Evidence Items: {len(all_evidence)}")
            
            # Check for social media aggregated evidence
            social_media_evidence = [e for e in all_evidence if e.source_type and 'social' in e.source_type.lower()]
            if social_media_evidence:
                print(f"   � Social Media Evidence Found: {len(social_media_evidence)} items")
                for j, evidence in enumerate(social_media_evidence, 1):
                    print(f"      {j}. Type: {evidence.source_type}, Quality: {evidence.quality}")
                    if hasattr(evidence, 'takeaways') and evidence.takeaways:
                        preview = evidence.takeaways[:150] + "..." if len(evidence.takeaways) > 150 else evidence.takeaways
                        print(f"         Content: {preview}")
            
            other_evidence = [e for e in all_evidence if not (e.source_type and 'social' in e.source_type.lower())]
            if other_evidence:
                print(f"   📰 Other Evidence: {len(other_evidence)} items")
                for j, evidence in enumerate(other_evidence[:3], 1):  # Show first 3
                    action_name = evidence.action.name if evidence.action else "Unknown"
                    print(f"      {j}. Action: {action_name}")
            
            if hasattr(report, 'reasoning') and report.reasoning:
                reasoning = report.reasoning[:150] + "..." if len(report.reasoning) > 150 else report.reasoning
                print(f"   🤔 Reasoning: {reasoning}")
                
        except Exception as e:
            print(f"❌ Error during fact-checking: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎉 DEFAME Fact-Checking Test Complete!")

def test_fact_check_sync():
    """Run the fact-checking test"""
    return test_fact_check_with_claim()

if __name__ == "__main__":
    test_fact_check_sync()

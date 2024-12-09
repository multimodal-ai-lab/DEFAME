from infact.fact_checker import FactChecker

fc = FactChecker(llm="gpt_4o", print_log_level="debug")

for i in range(1000):
    content = input("Enter a claim to be fact-checked: ")

    doc = fc.check_content(content)
    fc.logger.save_fc_doc(doc, claim_id=i)

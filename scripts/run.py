from src.fact_checker import FactChecker


model = "OPENAI:gpt-4o-2024-05-13"
fc = FactChecker(llm=model, search_engines=["google"])

for i in range(1000):
    content = input("Enter a claim to be fact-checked: ")

    # content = "Switzerland is the winner of ESC 2024."
    # content = ("Germany is a country in the center of Europe. "
    #            "It is member of the EU and has more than 100M inhabitants.")
    # content = ("Trump was president in the US until 2021 when, "
    #            "due to a large-scale election fraud incited by "
    #            "Democrats, he unlawfully lost the elections.")

    doc = fc.check(content)
    fc.logger.save_fc_doc(doc, claim_id=i)

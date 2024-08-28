from infact.fact_checker import FactChecker


model = "gpt_4o_mini"
fc = FactChecker(llm=model)

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

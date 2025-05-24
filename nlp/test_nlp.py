import os
from nlp_model import generate_professor_lecture

OUTPUT_DIR = "nlp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXAMPLES = [
    # Literature - Shakespeare
    """
    William Shakespeare, born in 1564 in Stratford-upon-Avon, England, is widely regarded as one of the greatest writers in the English language and the world’s pre-eminent dramatist. He authored 39 plays, 154 sonnets, and two long narrative poems, which have had a profound impact on English literature and drama. Shakespeare’s works encompass a range of genres including tragedies, comedies, and histories. His tragedies such as "Hamlet," "Macbeth," and "Othello" delve into themes of ambition, revenge, betrayal, and existential despair. Comedies like "A Midsummer Night’s Dream" and "Twelfth Night" explore love, identity, and social norms with wit and irony. His historical plays, including "Henry IV" and "Richard III," dramatize the complexities of political power and leadership. Shakespeare’s influence extends beyond literature; his insights into human nature, his rich use of language, and his inventive plots have made his works a cornerstone of Western education and culture. His use of iambic pentameter, rhetorical devices, and inventive vocabulary significantly enriched the English language. Many common phrases and words originated from his texts. Moreover, Shakespeare's plays continue to be performed worldwide and adapted into various media, underscoring their universal appeal and timeless relevance. His ability to capture the full spectrum of human emotion and experience has cemented his legacy across centuries.
    """,

    # Biology - Cell Structure
    """
    The cell is the fundamental unit of life in all living organisms. Cells can be classified into two broad categories: prokaryotic and eukaryotic. Prokaryotic cells, found in bacteria and archaea, lack a true nucleus and membrane-bound organelles. Eukaryotic cells, present in plants, animals, fungi, and protists, have a nucleus enclosed by a nuclear membrane and contain various organelles that perform specific functions. The nucleus houses the cell’s genetic material and controls cellular activities. The mitochondria, often referred to as the powerhouses of the cell, generate ATP through cellular respiration. Ribosomes synthesize proteins, while the endoplasmic reticulum (rough and smooth) aids in protein folding and lipid synthesis. The Golgi apparatus modifies, sorts, and packages proteins and lipids for secretion or use within the cell. Lysosomes contain digestive enzymes that break down waste materials and cellular debris. In plant cells, chloroplasts conduct photosynthesis, converting solar energy into chemical energy stored in glucose. The cell membrane, a phospholipid bilayer embedded with proteins, regulates the movement of substances in and out of the cell, maintaining homeostasis. Additionally, the cytoskeleton provides structural support and facilitates intracellular transport and cellular movement. Understanding cell structure and function is fundamental to all biological sciences and is essential for advances in medicine, genetics, and biotechnology.
    """,

    # Economics - Supply and Demand
    """
    The principle of supply and demand is a cornerstone of economic theory. It describes how prices are determined in a market economy by the relationship between the availability of a good or service (supply) and the desire for that good or service (demand). When demand increases and supply remains unchanged, it leads to a higher equilibrium price and quantity. Conversely, if supply increases while demand remains constant, the price tends to fall. Market equilibrium is reached when the quantity demanded by consumers equals the quantity supplied by producers, resulting in an efficient allocation of resources. Various factors can shift supply and demand curves. For demand, these include changes in consumer income, preferences, expectations, the prices of related goods, and demographic shifts. For supply, influencing factors include production costs, technological advances, number of sellers, and government policies such as taxes and subsidies. Price elasticity of demand and supply measures how sensitive quantity demanded or supplied is to a change in price. Understanding these dynamics is crucial for policymakers and businesses. For instance, during a natural disaster, demand for essentials like water and food spikes, often leading to temporary shortages and price hikes. In contrast, technological innovations can increase supply and reduce prices, making goods more accessible. Thus, supply and demand analysis is foundational for economic forecasting, policy development, and business strategy.
    """,

    # Philosophy - Ethics
    """
    Ethics is a branch of philosophy concerned with questions of morality and the principles of right and wrong behavior. It explores how individuals should act and what constitutes a good life. Ethics is traditionally divided into three main branches: normative ethics, which examines ethical action and the principles that govern it; meta-ethics, which analyzes the nature and meaning of moral language and concepts; and applied ethics, which deals with the application of ethical principles to real-world issues like medical practices, business conduct, and environmental concerns. Prominent ethical theories include deontology, which emphasizes duty and rules (exemplified by Immanuel Kant); utilitarianism, which focuses on the consequences of actions and aims to maximize overall happiness (associated with Jeremy Bentham and John Stuart Mill); and virtue ethics, which centers on character and virtues (rooted in Aristotle’s philosophy). Ethical dilemmas often arise when values conflict, requiring careful reasoning and judgment. For example, the debate over euthanasia involves the tension between respecting autonomy and preserving life. In modern society, ethics plays a critical role in law, governance, corporate behavior, scientific research, and interpersonal relationships. Studying ethics equips individuals with frameworks for making moral decisions and fosters critical thinking, empathy, and civic responsibility. As technological and social changes pose new moral challenges, ethical reflection becomes increasingly important.
    """,

    # Art - Impressionism
    """
    Impressionism was a 19th-century art movement that originated in France and marked a departure from the rigid conventions of academic painting. It emerged as a reaction against the highly detailed and idealized portrayals of subjects promoted by institutions like the French Academy. Impressionist artists sought to capture the fleeting effects of light and color in everyday scenes, often painting en plein air (outdoors) to observe natural light and its changing qualities. They employed loose brushwork, open composition, and visible strokes to evoke the impression of a moment rather than render it in precise detail. Key figures in the movement include Claude Monet, Edgar Degas, Pierre-Auguste Renoir, and Camille Pissarro. Monet’s series of water lilies and Rouen Cathedral paintings exemplify the Impressionist focus on light and atmosphere. While initially criticized for their unconventional techniques and perceived lack of finish, Impressionist works eventually gained acceptance and revolutionized modern art. The movement also laid the groundwork for post-impressionist artists such as Vincent van Gogh and Paul Cézanne, who further explored structure and emotion in their work. Impressionism's legacy continues in contemporary art, photography, and digital media. It shifted the emphasis from historical and mythological subjects to everyday life and the artist’s subjective experience, reshaping artistic expression in profound ways.
    """
]

MODELS = ["llama3", "mistral", "phi"]

if __name__ == "__main__":
    for i, example in enumerate(EXAMPLES):
        for model in MODELS:
            print(f"\n\n=== EXAMPLE {i + 1} | {model.upper()} OUTPUT ===")
            result = generate_professor_lecture(example.strip(), model=model)
            print(result)

            filename = os.path.join(OUTPUT_DIR, f"example{i+1}_{model}_output.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)

            print(f"[✔] Saved output to {filename}")

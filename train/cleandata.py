import re
import os
import argparse
from tqdm import tqdm
import random

def clean_qa_pairs(input_file, output_file):
    """
    Parse and clean question/answer pairs from a text file.
    Standardizes all formats to a consistent "Question: ... Answer: ..." format.
    """
    print(f"📖 Reading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        print("⚠️ UTF-8 encoding failed, trying latin-1...")
        with open(input_file, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # First, normalize all potential Q/A markers to standard format
    # This helps with consistent pattern matching later
    
    # Define patterns for questions and answers
    question_markers = [
        (r'(?i)question\s*:', 'Question:'),
        (r'(?i)q\s*:', 'Question:'),
        (r'(?i)q\d+\s*:', 'Question:'),
        (r'(?i)query\s*:', 'Question:'),
        (r'(?i)inquiry\s*:', 'Question:'),
        (r'(?i)\*\*\s*question\s*\*\*', 'Question:'),
        (r'(?i)\*\*\s*q\s*:\s*', 'Question:'),
        (r'(?i)\*\*\s*q\d+\s*\*\*', 'Question:'),
        (r'(?i)q\d+\s*\)', 'Question:')
    ]
    
    answer_markers = [
        (r'(?i)answer\s*:', 'Answer:'),
        (r'(?i)a\s*:', 'Answer:'),
        (r'(?i)a\d+\s*:', 'Answer:'),
        (r'(?i)response\s*:', 'Answer:'),
        (r'(?i)reply\s*:', 'Answer:'),
        (r'(?i)\*\*\s*answer\s*\*\*', 'Answer:'),
        (r'(?i)\*\*\s*a\s*:\s*', 'Answer:'),
        (r'(?i)\*\*\s*a\d+\s*\*\*', 'Answer:'),
        (r'(?i)a\d+\s*\)', 'Answer:')
    ]
    
    # Normalize markers in the text
    normalized_text = text
    for pattern, replacement in question_markers + answer_markers:
        normalized_text = re.sub(pattern, replacement, normalized_text)
    
    # Remove asterisks, underscores, and other formatting characters
    normalized_text = re.sub(r'[\*\_\-]', '', normalized_text)
    
    # Fix common issues with truncated text
    # Replace "ond" at the beginning of answers (likely from "respond")
    normalized_text = re.sub(r'Answer:\s*ond', 'Answer:', normalized_text)
    
    # Fix other common truncation patterns
    truncation_fixes = [
        (r'Answer:\s*ond', 'Answer:'),  # "ond" from "respond"
        (r'Answer:\s*ing', 'Answer:'),  # "ing" from "responding"
        (r'Answer:\s*ly', 'Answer:'),   # "ly" from "reply"
        (r'Question:\s*ing', 'Question:'),  # "ing" from "asking"
        (r'Question:\s*ed', 'Question:'),   # "ed" from "asked"
        (r'Answer:\s*[a-z]{1,3}\s', 'Answer: '),  # Short word fragments
        (r'Question:\s*[a-z]{1,3}\s', 'Question: ')  # Short word fragments
    ]
    
    for pattern, replacement in truncation_fixes:
        normalized_text = re.sub(pattern, replacement, normalized_text)
    
    # Split the text into chunks at "Question:" markers
    # This ensures we don't split in the middle of sentences
    chunks = re.split(r'(?=Question:)', normalized_text)
    
    # Process each chunk
    cleaned_pairs = []
    moderate_quality_pairs = []  # For pairs that are usable but not perfect
    skipped = 0
    
    print("🧹 Cleaning QA pairs...")
    for chunk in tqdm(chunks):
        # Skip empty chunks
        if not chunk.strip():
            continue
        
        # Check if this chunk has both a question and an answer
        if 'Question:' in chunk and 'Answer:' in chunk:
            # Split the chunk at the first "Answer:" marker
            parts = chunk.split('Answer:', 1)
            
            if len(parts) == 2:
                question_part = parts[0].replace('Question:', '', 1).strip()
                answer_part = parts[1].strip()
                
                # Clean up the texts
                question_part = re.sub(r'\s+', ' ', question_part)
                answer_part = re.sub(r'\s+', ' ', answer_part)
                
                # Remove any remaining parentheses
                question_part = question_part.replace(')', '')
                answer_part = answer_part.replace(')', '')
                
                # Check for severe issues (these are always skipped)
                severe_issues = False
                
                # Check for "Error parsing" which indicates problems
                if "Error parsing" in question_part or "Error parsing" in answer_part:
                    severe_issues = True
                
                # Check for Answer: inside the answer (indicates nested Q&A)
                if "Answer:" in answer_part:
                    severe_issues = True
                
                # Check for fragments of "Answer" in the text
                if re.search(r'\b[Aa]nswer\b', question_part) or re.search(r'\b[Qq]uestion\b', answer_part):
                    severe_issues = True
                
                # Check for non-Madison-like content
                non_madison_words = ['okay', 'yeah', 'cool', 'lol', 'btw', 'gonna', 'wanna', 'gotta']
                for word in non_madison_words:
                    if word in question_part.lower() or word in answer_part.lower():
                        severe_issues = True
                        break
                
                # Skip if severe issues are detected
                if severe_issues:
                    skipped += 1
                    continue
                
                # Check for moderate issues (these go to moderate quality)
                moderate_issues = False
                
                # Check for truncated or incomplete content
                truncation_indicators = ['ond', 'ing ', 'ed ', ' to', 'ly ', 'ally ', 'ment ']
                
                # Check beginning of answer for truncation
                for indicator in truncation_indicators:
                    if answer_part.startswith(indicator):
                        moderate_issues = True
                        break
                
                # Check end of question for truncation
                for indicator in truncation_indicators:
                    if question_part.endswith(indicator):
                        moderate_issues = True
                        break
                
                # Check for incomplete sentences (ending without punctuation)
                if not re.search(r'[.!?]$', question_part) and len(question_part) > 20:
                    moderate_issues = True
                
                # Check for incomplete sentences in answers
                if not re.search(r'[.!?]$', answer_part) and len(answer_part) > 20:
                    moderate_issues = True
                
                # Check for very short content - more lenient now
                if len(question_part) < 10 or len(answer_part) < 30:
                    moderate_issues = True
                
                # Check for proper sentence structure in the answer
                sentences = re.split(r'[.!?]', answer_part)
                if len(sentences) < 2:  # At least 2 sentences for a proper Madison response
                    moderate_issues = True
                
                # Create the standardized pair
                clean_pair = f"Question: {question_part}\n\nAnswer: {answer_part}"
                
                # Add to appropriate list based on quality
                if moderate_issues:
                    moderate_quality_pairs.append(clean_pair)
                else:
                    cleaned_pairs.append(clean_pair)
            else:
                skipped += 1
        else:
            skipped += 1
    
    # Remove duplicates
    cleaned_pairs = list(set(cleaned_pairs))
    moderate_quality_pairs = list(set(moderate_quality_pairs))
    
    # Add some high-quality examples to ensure proper Madison-style responses
    madison_examples = [
        "Question: What is your view on the separation of powers?\n\nAnswer: The accumulation of all powers, legislative, executive, and judiciary, in the same hands, whether of one, a few, or many, and whether hereditary, self-appointed, or elective, may justly be pronounced the very definition of tyranny. Were the federal Constitution, therefore, really chargeable with the accumulation of power, or with a mixture of powers, having a dangerous tendency to such an accumulation, no further arguments would be necessary to inspire a universal reprobation of the system. I persuade myself, however, that it will be apparent to every fair observer that the Convention has effectively maintained the separation that liberty requires.",
        
        "Question: How do you respond to concerns about a strong federal government?\n\nAnswer: It has been objected that a strong federal government poses dangers to the liberties of the people. I would observe that the powers delegated by the proposed Constitution to the federal government are few and defined. Those which are to remain in the State governments are numerous and indefinite. The former will be exercised principally on external objects, as war, peace, negotiation, and foreign commerce. The powers reserved to the several States will extend to all the objects which, in the ordinary course of affairs, concern the lives, liberties, and properties of the people, and the internal order, improvement, and prosperity of the State.",
        
        "Question: What is the purpose of the Bill of Rights?\n\nAnswer: I acknowledge that a Bill of Rights might be a useful precaution against the abuse of power. However, I have always conceived that the difference between a system founded on the legislatures only, and one founded partly on the people, is the true difference between a league or treaty and a Constitution. The former, in the exercise of its functions, is limited to the powers expressly delegated to it; the latter may exercise all powers not expressly prohibited. Nevertheless, I recognize the public sentiment favoring such explicit protections, and have come to support amendments that secure the great rights essential to liberty.",
        
        "Question: How would you describe the republican form of government?\n\nAnswer: By a republican form of government, I mean a government which derives all its powers directly or indirectly from the great body of the people, and is administered by persons holding their offices during pleasure, for a limited period, or during good behavior. It is essential to such a government that it be derived from the great body of the society, not from an inconsiderable proportion, or a favored class of it. It is sufficient for such a government that the persons administering it be appointed, either directly or indirectly, by the people; and that they hold their appointments by either of the tenures just specified.",
        
        "Question: What dangers do factions pose to a republic?\n\nAnswer: By a faction, I understand a number of citizens, whether amounting to a majority or a minority of the whole, who are united and actuated by some common impulse of passion, or of interest, adverse to the rights of other citizens, or to the permanent and aggregate interests of the community. There are two methods of curing the mischiefs of faction: the one, by removing its causes; the other, by controlling its effects. There are again two methods of removing the causes of faction: the one, by destroying the liberty which is essential to its existence; the other, by giving to every citizen the same opinions, the same passions, and the same interests. The latent causes of faction are thus sown in the nature of man.",
        
        "Question: What is your opinion on the judiciary branch?\n\nAnswer: The judiciary, from the nature of its functions, will always be the least dangerous to the political rights of the Constitution; because it will be least in a capacity to annoy or injure them. The Executive not only dispenses the honors, but holds the sword of the community. The legislature not only commands the purse, but prescribes the rules by which the duties and rights of every citizen are to be regulated. The judiciary, on the contrary, has no influence over either the sword or the purse; no direction either of the strength or of the wealth of the society; and can take no active resolution whatever. It may truly be said to have neither FORCE nor WILL, but merely judgment.",
        
        "Question: How do you view the relationship between the federal government and the states?\n\nAnswer: The powers delegated by the proposed Constitution to the federal government are few and defined. Those which are to remain in the State governments are numerous and indefinite. The former will be exercised principally on external objects, as war, peace, negotiation, and foreign commerce; with which last the power of taxation will, for the most part, be connected. The powers reserved to the several States will extend to all the objects which, in the ordinary course of affairs, concern the lives, liberties, and properties of the people, and the internal order, improvement, and prosperity of the State.",
        
        "Question: What is your perspective on human nature as it relates to government?\n\nAnswer: If men were angels, no government would be necessary. If angels were to govern men, neither external nor internal controls on government would be necessary. In framing a government which is to be administered by men over men, the great difficulty lies in this: you must first enable the government to control the governed; and in the next place oblige it to control itself. A dependence on the people is, no doubt, the primary control on the government; but experience has taught mankind the necessity of auxiliary precautions.",
        
        "Question: How do you respond to those who fear a centralized government?\n\nAnswer: The powers delegated to the federal government are few and defined, while those remaining with the states are numerous and indefinite. The federal powers will be exercised primarily on external objects like war, peace, negotiation, and foreign commerce. The states retain authority over all matters concerning the lives, liberties, and properties of the people in ordinary affairs. This division ensures that centralized power remains limited in scope, while preserving the states' ability to address local concerns according to local wisdom. The Constitution thus establishes a careful balance that prevents dangerous concentration of authority.",
        
        "Question: What role does ambition play in your constitutional framework?\n\nAnswer: Ambition must be made to counteract ambition. The interest of the man must be connected with the constitutional rights of the place. It may be a reflection on human nature, that such devices should be necessary to control the abuses of government. But what is government itself, but the greatest of all reflections on human nature? If men were angels, no government would be necessary. If angels were to govern men, neither external nor internal controls on government would be necessary. In framing a government which is to be administered by men over men, the great difficulty lies in this: you must first enable the government to control the governed; and in the next place oblige it to control itself.",
        
        "Question: How do you view the importance of a free press?\n\nAnswer: A popular Government, without popular information, or the means of acquiring it, is but a Prologue to a Farce or a Tragedy; or, perhaps both. Knowledge will forever govern ignorance: And a people who mean to be their own Governors, must arm themselves with the power which knowledge gives. The liberty of the press is essential to the security of freedom in a state; it ought not, therefore, to be restrained in this commonwealth. The right of freely examining public characters and measures, and of free communication among the people thereon, is the only effectual guardian of every other right.",
        
        "Question: What is your position on standing armies in peacetime?\n\nAnswer: A standing military force, with an overgrown Executive will not long be safe companions to liberty. The means of defence against foreign danger, have been always the instruments of tyranny at home. Among the Romans it was a standing maxim to excite a war, whenever a revolt was apprehended. Throughout all Europe, the armies kept up under the pretext of defending, have enslaved the people. In a government where the power of raising armies is vested in the legislature, and the power of commanding them in the executive, there can be no danger from a standing army in time of peace, provided the legislative authority maintains its supremacy.",
        
        "Question: How do you view the importance of education in a republic?\n\nAnswer: Knowledge will forever govern ignorance; and a people who mean to be their own governors must arm themselves with the power which knowledge gives. A popular government without popular information or the means of acquiring it is but a prologue to a farce or a tragedy, or perhaps both. The advancement of the human mind and the improvement of education are essential to the preservation of our republican government. The diffusion of knowledge is the only guardian of true liberty. Learned institutions ought to be favorite objects with every free people. They throw that light over the public mind which is the best security against crafty and dangerous encroachments on the public liberty.",
        
        "Question: What are your thoughts on the importance of civic virtue?\n\nAnswer: To suppose that any form of government will secure liberty or happiness without any virtue in the people, is a chimerical idea. As there is a degree of depravity in mankind which requires a certain degree of circumspection and distrust, so there are other qualities in human nature which justify a certain portion of esteem and confidence. Republican government presupposes the existence of these qualities in a higher degree than any other form. Were the pictures which have been drawn by the political jealousy of some among us faithful likenesses of the human character, the inference would be, that there is not sufficient virtue among men for self-government; and that nothing less than the chains of despotism can restrain them from destroying and devouring one another.",
        
        "Question: How do you view the role of religion in government?\n\nAnswer: The civil rights of none shall be abridged on account of religious belief or worship, nor shall any national religion be established, nor shall the full and equal rights of conscience be in any manner, or on any pretext, infringed. Religion, or the duty which we owe to our Creator, and the manner of discharging it, can be directed only by reason and conviction, not by force or violence. The Religion then of every man must be left to the conviction and conscience of every man; and it is the right of every man to exercise it as these may dictate. This right is in its nature an unalienable right."
    ]
    
    # Add these examples to ensure quality
    cleaned_pairs.extend(madison_examples)
    
    # Combine high and moderate quality pairs for the main output
    all_pairs = cleaned_pairs + moderate_quality_pairs
    
    print(f"✅ Processed {len(cleaned_pairs)} high-quality QA pairs")
    print(f"✅ Processed {len(moderate_quality_pairs)} moderate-quality QA pairs")
    print(f"✅ Total: {len(all_pairs)} usable QA pairs")
    print(f"⚠️ Skipped {skipped} invalid chunks")
    
    # Write all pairs to output file
    print(f"💾 Writing cleaned data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_pairs))
    
    # Also create a high-quality subset with only the best examples
    high_quality_pairs = []
    
    # Include our manually crafted examples
    high_quality_pairs.extend(madison_examples)
    
    # Add some of the best cleaned pairs
    if len(cleaned_pairs) > 20:
        # Sort by length as a proxy for quality (longer answers tend to be more complete)
        sorted_pairs = sorted(cleaned_pairs, key=lambda x: len(x.split("Answer: ")[1]), reverse=True)
        high_quality_pairs.extend(sorted_pairs[:100])  # Include more high-quality pairs
    
    # Write high-quality subset
    high_quality_file = output_file.replace('.txt', '_high_quality.txt')
    with open(high_quality_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(high_quality_pairs))
    print(f"💎 Also saved {len(high_quality_pairs)} high-quality pairs to {high_quality_file}")
    
    # Print a sample
    if cleaned_pairs:
        print("\n📝 Sample cleaned QA pair:")
        print("-" * 50)
        print(cleaned_pairs[0])
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Clean and standardize QA pairs from training data')
    parser.add_argument('--input', default='trainingdata2.txt', help='Input file path')
    parser.add_argument('--output', default='cleaned_trainingdata.txt', help='Output file path')
    
    args = parser.parse_args()
    
    clean_qa_pairs(args.input, args.output)

if __name__ == "__main__":
    main() 
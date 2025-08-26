import random


RPS_Conversion = {"rock": 0, "paper": 1, "scissors": 2}
reverse_RPS = {v: k for k, v in RPS_Conversion.items()}


def get_user_choice(choice):
    # Takes user choice as input
    return RPS_Conversion.get(choice.lower(), None)    

def get_computer_choice(): 
    #Generates random choice from RPS Conversion
    return random.choice(list(RPS_Conversion.values()))

def determine_winner(user, comp):
    if user == comp:
        return 0   # draw
    elif (user - comp) % 3 == 1:
        return 1   # user wins
    else:
        return -1  # user loses
    
def main_game():
    while True:
        raw_choice =  input("Enter rock, paper, or scissors: ")
        user_choice = get_user_choice(raw_choice)
        if user_choice is None:
            print("Invalid choice, try again.")
            continue #If program gets unexpected value retry
        comp_choice = get_computer_choice()
        # Converts results from int to string to print output
        result_text = {1: "You win!", 0: "It's a draw!", -1: "You lose!"}
        outcome = determine_winner(user_choice, comp_choice)
        # Gives user clean output for info what everyone played and result!
        print(f"They played {reverse_RPS[comp_choice]} You played {reverse_RPS[user_choice]} \n{result_text[outcome]}")
        play_again = input("Play again? (y/n): ").lower()
        if play_again != "y":
            break

main_game()



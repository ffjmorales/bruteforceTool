import time
import string
import itertools

def brute_force_crack(password, charset, max_length):
    attempts = 0
    start = time.time()

    for length in range(1, max_length + 1):
        for guess in itertools.product(charset, repeat=length):
            guess_str = ''.join(guess)
            attempts += 1
            if guess_str == password:
                end = time.time()
                time_taken = end - start

                # If the time is very small, treat it as "instantaneous"
                if time_taken == 0:
                    guesses_per_second = float('inf')  # Indicates it was too fast
                else:
                    guesses_per_second = attempts / time_taken  # Actual guesses per second rate

                return {
                    "password": guess_str,
                    "attempts": attempts,
                    "time_taken": time_taken,
                    "guesses_per_second": guesses_per_second
                }

    return None

if __name__ == "__main__":
    password = "aaa"  # Password to test
    charset = string.ascii_lowercase + string.digits
    max_length = 3

    print(f"Cracking password: {password}")
    result = brute_force_crack(password, charset, max_length)

    if result:
        print(f"✅ Password Found: {result['password']}")
        print(f"🔢 Tries: {result['attempts']}")
        print(f"⏱️ Time: {result['time_taken']:.4f} second")

        # Verify and display guesses per second rate
        guesses_per_second = result.get('guesses_per_second', 0)
        if guesses_per_second == float('inf'):
            print("⚙️ Real Time: Instant (too fast!)")
        else:
            print(f"⚙️ Real Time: {int(guesses_per_second)} tries/second")
    else:
        print("❌ Password was not found in the given length limit.")

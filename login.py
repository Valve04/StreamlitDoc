# import streamlit as st
# import streamlit_authenticator as stauth

# # User credentials
# names = ["Admin"]
# usernames = ["admin"]
# passwords = ["password123"]

# # Hash passwords
# hasher = stauth.Hasher(passwords)  # Initialize hasher with passwords list
# hashed_passwords = []
# for password in passwords:
#     hashed_password = hasher.hash(password)  # Hash each password individually
#     hashed_passwords.append(hashed_password)

# # Create credentials dictionary
# credentials = {
#     "usernames": {}
# }

# # Populate credentials dictionary
# for (name, username, hashed_password) in zip(names, usernames, hashed_passwords):
#     credentials["usernames"][username] = {
#         "name": name,
#         "password": hashed_password
#     }

# # Create the authenticator object
# authenticator = stauth.Authenticate(
#     credentials,
#     "streamlit_app_cookie",
#     "random_key",
#     cookie_expiry_days=30
# )

# # Create login widget - Place it in sidebar
# name, authentication_status, username = authenticator.login("Login", location="main")

# # Handle authentication status
# if authentication_status:
#     st.success(f"Welcome, {name}!")
#     if st.button("Logout"):
#         authenticator.logout("Logout", location="main")
# elif authentication_status == False:
#     st.error("Invalid username or password")
# elif authentication_status == None:
#     st.warning("Please enter your username and password")





import streamlit as st
import app  # Ensure this module exists and has an app() function

# Dictionary to store user credentials (for demo purposes only)
users = {"Admin": "Admin", "xxx": "xxx"}

# Function to authenticate users
def authenticate(username, password):
    if username in users and users[username] == password:
        return True
    return False

# Streamlit login page
def login():
    st.title("Login Page")

    # Input fields for username and password
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")

    # Login button
    login_button = st.button("Login")

    # Check if login button was clicked and if credentials are correct
    if login_button:
        if authenticate(username, password):
            st.success("Login successful!")
            st.session_state["authenticated"] = True
            st.session_state["page"] = "Main Page"  # Navigate to the main page
        else:
            st.error("Invalid username or password")


# Streamlit logout page
def logout():
    st.session_state["authenticated"] = False
    st.session_state["page"] = "Login"
    st.success("You have been logged out!")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "page" not in st.session_state:
    st.session_state["page"] = "Login"

# Page navigation based on authentication status
if st.session_state["authenticated"]:
    if st.session_state["page"] == "Main Page":
        app.Hello()  # Call the app function when authenticated
        
        # Add a logout button on the main page
        if st.button("Logout"):
            logout()  # Call the logout function
else:
    login()  # Show login page if not authenticated

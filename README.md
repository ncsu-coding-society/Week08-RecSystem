# Week08-RecSystem

0. Pre-Setup
Before coding, clone this repository and create a branch with your name (mine would be called 'preeti'). Push this branch to origin before starting to code. You can work in app.py and commit + push all your changes to your branch - don't merge with main!


Assume {name} is the name of the branch.

To create and push a new branch:
```
git branch {name}
git checkout {name}
git push -u origin {name}
```

To commit and push changes:
```
git add -A
git add app.py
git commit -m "commit message"
git push origin
```

To delete branch:
```
git branch -d {name}
git push origin –delete {name}
```

1. Setup

Install Dependencies:
```
pip3 install requests pandas torch numpy transformers scikit-learn
```

Setup Environment Variables:
```
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
```

2. Ingest Data

3. Compute Embeddings

4. Create Recommendation System

5. Test System + Create Interface

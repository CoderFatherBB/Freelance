import subprocess
import time
import sys
import os

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

def main():
    print("Fetching untracked and modified files...")
    # Get modified files
    res = run_cmd("git ls-files -m")
    modified = [f for f in res.stdout.split('\n') if f.strip()]
    
    # Get untracked files
    res = run_cmd("git ls-files --others --exclude-standard")
    untracked = [f for f in res.stdout.split('\n') if f.strip()]
    
    all_files = list(set(modified + untracked))
    
    if not all_files:
        print("No files to push.")
        return
    
    print(f"Total files to commit: {len(all_files)}")
    
    batch_size = 100
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i+batch_size]
        print(f"--- Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size} ---")
        
        # Add files safely
        for file in batch:
            subprocess.run(['git', 'add', file])
            
        commit_msg = f"Batch commit {i//batch_size + 1}"
        commit_res = subprocess.run(['git', 'commit', '-m', commit_msg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if commit_res.returncode == 0:
            print(f"Committed {len(batch)} files.")
        else:
            print(f"Nothing to commit or error: {commit_res.stderr}")
            
        push_res = subprocess.run(['git', 'push', 'origin', 'main'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if push_res.returncode == 0:
            print(f"Pushed batch {i//batch_size + 1} successfully.")
        else:
            print(f"Failed to push batch {i//batch_size + 1}: {push_res.stderr}")
            print("Stopping...")
            sys.exit(1)
            
        # Sleep to be gentle on the remote server
        time.sleep(1)
        
    print("All batches pushed successfully!")

if __name__ == '__main__':
    main()

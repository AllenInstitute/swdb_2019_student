contributing to swdb_2019_tools
===============================

your first contribution
-----------------------

## 1. Clone this repository. 
 - **how:** `git clone https://github.com/alleninstitute/swdb_2019_tools`
 - **why:** This will download the repository to your computer so you can add and edit files!
 - **did it work?** you should have a directory named `swdb_2019_tools` in your working directory. e.g.

        $ ls
        swdb_2019_tools

## 2. navigate to your clone
- **how:**

        $ cd swdb_2019_tools
- **why:** in order to apply git commands to your clone, you need to be working in your clone's directory.
- **did it work?** running `ls` ought to show you something like:

    ```
    LICENSE.txt  README.md
    ```

## 3. check your clone's git status
- **how:**

        $ git status
- **why:** This tells you what branch you are working on, whether it is up to date with the version on Github, and what changes you have made locally. You dont
- **did it work?** You should see:

        On branch master
        Your branch is up to date with 'origin/master'.

        nothing to commit, working tree clean

## 4. create a branch to work on
- **how:** 

        git checkout -b {your_name_here}s_tutorial_branch
- **why:** Lots of people are trying to add code or make changes to the master branch at the same time. This would be pretty chaotic if people were actively typing over each other. We can cut down on the confusion by getting discrete chunks of work done in our own workspaces, then merging them together.
- **did it work?** if you run `git status` you ought to see something like:

        On branch {your_name_here}s_tutorial_branch
        nothing to commit, working tree clean


## 5. make a change
- **how:** Try adding a file, using an editor of your choice. For the purpose of this exercise a unique name would be sensible (so that you don't conflict with others). An example (should work on Linux / OSX):

        echo "hello world" >> {your name here}s_tutorial_file.txt

- **why:** Making changes is the whole point!
- **did it work?** running `git status` ought to notify you of a new untracked file

        On branch {your name here}s_tutorial_branch
        Untracked files:
        (use "git add <file>..." to include in what will be committed)

                {your name here}s_tutorial_file.txt

        nothing added to commit but untracked files present (use "git add" to track)

## 6. add your changes to the index
- **how:** `git add {your name here}s_tutorial_file.txt`
- **why:** Your new file is untracked. If you were to commit now, it would not be included in the commit.
- **did it work?** As per usual, run `git status`. You should see something like:

        On branch {your name here}s_tutorial_branch
        Changes to be committed:
        (use "git reset HEAD <file>..." to unstage)

                new file:   {your name here}s_tutorial_file.txt

## 7. commit your change
- **how:** `git commit -m "adds my tutorial file"`
- **why:** A git commit defines a saved state of the repository. You can push them to or pull them from remotes, merge them across branches, and look them up or revert to them. In order for your work to be tracked and shared, it needs to be committed.
- **did it work?** `git status` should show no changes (since the last commit!)

        On branch {your_name_here}s_tutorial_branch
        nothing to commit, working tree clean

    the top of `git log` should show a new latest commit:

        commit a8fa0e21576b9884a94f1aad41d5c9e0fdcbb320 (HEAD -> niles_branch)
        Author: nile graddis <nilegraddis@gmail.com>
        Date:   Tue Aug 20 15:45:58 2019 -0700

                look, a commit message

## 8. push your change
- **how:** `git push origin {your name here}s_tutorial_branch`
- **why:** In order for other people to use your changes, you will need to upload them to github.
- **did it work?** go to https://github.com/alleninstitute/swdb_2019_tools. You should see a little yellow banner with your branch name on it and a button for creating a new pull request.

## 9. make a pull request
- **how:** On https://github.com/alleninstitute/swdb_2019_tools, click on "pull requests", then "new pull request".
- **why:** Your ultimate goal is to get your work into the master branch so that others who clone swdb_2019_tools can make use of your code. Doing so via a pull request gives others a chance to review your changes.
- **did it work?** if you go to https://github.com/alleninstitute/swdb_2019_tools/pulls, you should see your pull request in the list

advanced topics
---------------
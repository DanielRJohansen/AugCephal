#pragma once


#include <iostream>

using namespace std;



class UnorderedIntTree {
	struct Node;
public:	
	UnorderedIntTree() {}
	bool addVal(int key) {
		if (key < 0) {
			printf("Inserting illegal key: %d\n\n",key);
			exit(-2);
		}
			
		if (root != NULL)
			if (root->addVal(key)) {
				treesize++;
				return true;
			}
		else {
			root = new Node(key);
			treesize++;
			return true;
		}			
		return false;
	}
	void deleteVal(int key) {				// ONLY CALL ON VALUES THAT DOES EXIST IN TREE!!!!!!!
		if (root == NULL) {
			//printf("ILLEGAL DELETION: root is NULL.\n");
		}
		else {
			if (root->find(key)) {				// Track whether the value is ACTUALLY found.
				if (root->deleteNode(key)) {	// return true only when root is value and root!
					delete(root);
					root = NULL;
				}
				treesize--;
			}
			//else
				//printf("ILLEGAL DELETION: key not found.\n");
		}
	}

	int* fetch() {
		if (root == NULL) {		// Illegal fetch
			return new int(-11);
		}
		int* index = new int(0);
		int* arr = new int[treesize];
		root->fetch(arr, index);

		delete(index);
		return arr;
	}
	void clear() {
		if (root != NULL)
			root->clear();
		delete(root);
		root = NULL;
	}
	void copy(UnorderedIntTree tree) {
		clear();
		int* values = tree.fetch();
		for (int i = 0; i < tree.size(); i++) {
			addVal(values[i]);
		}
		delete(values);
	}


	int size() { return treesize; }

	Node* root = NULL;
	int treesize = 0;
private:
	struct Node {
		Node() {}
		Node(int key) : value(key) {  }

		int value;
		bool leaf = true;
		Node* left = NULL;
		Node* right = NULL;


		bool addVal(int key) {
			if (value == key)
				return 0;

			Node* branch = getBranchPlaceholder(key);
			if (branch != NULL)
				return branch->addVal(key);			
			
			// Else
			branch = new Node(key);
			leaf = false;
			storeBranch(key, branch);	// assigns the object to correct left or right
			return 1;			
		}
		void fetch(int* arr, int* index) {
			if (left != NULL)
				left->fetch(arr, index);
			arr[*index] = value;
			*index += 1;
			if (right != NULL)
				right->fetch(arr, index);
		}
		
		bool find(int key) {
			if (key == value)
				return true;
			else if (key > value && right != NULL)
				return right->find(key);
			else if (key < value && left != NULL)
				return left->find(key);
			return false;
		}

		bool deleteNode(int key) {	
			if (key == value) {
				if (!leaf) {
					int replacement = findReplacement();
					deleteNode(replacement);	// Delete that val first, then set this element to that val. Called on this, beucase only a parent can delete a child!!!
					value = replacement;
				}
				else
					return true;									// In this case the parent will simply delete
			}
			else if (key > value) {
				if (right != NULL) {
					if (right->deleteNode(key)) {					// Check if right is key==val and leaf
						delete(right);
						right = NULL;
					}
				}													// No need for else case, the value does not exist in the tree
			}
			else if (key < value) {
				if (left != NULL) {
					if (left->deleteNode(key)) {
						delete(left);
						left = NULL;					
					}
				}
			}
			if (left == NULL && right == NULL) leaf = true;			// Update leaf status
			return false;
		}

		void clear() {
			if (left != NULL) {
				left->clear();
				delete(left);
				left = NULL;
			}
			if (right != NULL) {
				right->clear();
				delete(right);
				right = NULL;
			}
		}

		Node* getBranchPlaceholder(int key) {
			if (key > value)
				return right;
			else
				return left;
		}
		void storeBranch(int key, Node* branch) {
			if (key > value)
				right = branch;
			else
				left = branch;
		}

		int findReplacement() {
			if (left != NULL)		//Only called on non-leafs
				return left->max();
			else
				return right->min();
		}

		int min() {
			if (left != NULL)
				return left->min();
			return value;
		}
		int max() {
			if (right != NULL)
				return right->max();
			return value;
		}
	};
};
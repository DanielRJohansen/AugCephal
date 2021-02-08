#pragma once


#include <iostream>

using namespace std;


class UnorderedIntTree {
public:	
	UnorderedIntTree() {}
	void addVal(int key) {
		if (root != NULL)
			treesize += root->addVal(key);
		else {
			root = new Node(key);
			treesize++;
		}			
	}
	void deleteVal(int key) {				// ONLY CALL ON VALUES THAT DOES EXIST IN TREE!!!!!!!
		if (root != NULL) {
			if (root->deleteNode(key)) {
			delete(root);
			root = NULL;
			}
		}
		treesize--;							// Can NOT track wheter the value is ACTUALLY found.
	}

	int* fetch() {
		if (root == NULL)		// Illegal fetch
			return new int(-1);
		int* index = new int(0);
		//*index = 0;
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


	int size() { return treesize; }

	
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
		
		bool deleteNode(int key) {	
			if (key == value) {
				if (!leaf) {
					int replacement = findReplacement();
					deleteNode(replacement);	// Delete that val first, then set this element to that val. Called on this, beucase only a parent can delete a child!!!
					value = replacement;
				}
				else
					return true;			// Only case where the parent will delete
			}

			if (key > value) {
				if (right != NULL) {
					if (right->deleteNode(key)) {
						delete(right);
						right = NULL;
						if (left == NULL && right == NULL) leaf = true;
					}
				}						// No need for else case, the value does not exist in the tree
			}
			else if (key < value) {
				if (left != NULL) {
					if (left->deleteNode(key)) {
						delete(left);
						left = NULL;
						if (left == NULL && right == NULL) leaf = true;
					}
				}
			}
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


	Node* root = NULL;
	unsigned int treesize = 0;
};
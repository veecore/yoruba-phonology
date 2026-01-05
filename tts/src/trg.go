package main

func main() {
	var x any = "A"
	switch x.(type) {
		case string: println("string")
	}
}
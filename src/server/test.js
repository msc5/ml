// const WebSocket = require('ws')
//
// const socket = new WebSocket('ws://localhost:8765') // Replace with your WebSocket server URL
//
// // Event handler for when the WebSocket connection is established
// socket.onopen = () => {
//
// 	console.log('WebSocket connection established')
//
// 	// Send a message through the WebSocket
// 	const message = 'matthew'
// 	socket.send(message)
//
// }
//
// // Event handler for when a message is received from the WebSocket server
// socket.onmessage = (event) => {
//
// 	// const receivedMessage = JSON.parse(event.data)
// 	console.log('Received message:', String(event.data))
//
// }
//
// // Event handler for WebSocket connection closure
// socket.onclose = () => {
// 	console.log('WebSocket connection closed')
// }
//
// // Event handler for WebSocket connection errors
// socket.onerror = (error) => {
// 	console.error('WebSocket error:', error)
// }

// const XMLHttpRequest = require('xmlhttprequest')

const url = 'ws://localhost:8765' // Replace with your WebSocket server URL

// Function to send an Ajax request
function sendAjaxRequest(url, method, data) {
	return new Promise((resolve, reject) => {
		const xhr = new XMLHttpRequest()
		xhr.open(method, url)
		xhr.setRequestHeader('Content-Type', 'application/json')

		xhr.onload = () => {
			if (xhr.status >= 200 && xhr.status < 300) {
				resolve(xhr.response)
			} else {
				reject(new Error(`Request failed with status ${xhr.status}`))
			}
		}

		xhr.onerror = () => {
			reject(new Error('Request failed'))
		}

		// xhr.send(JSON.stringify(data))
		xhr.send(String(data))
	})
}

// Example usage
const message = 'matthew'

sendAjaxRequest(url, 'POST', message)
	.then((response) => {
		console.log('Ajax request successful:', response)
	})
	.catch((error) => {
		console.error('Ajax request failed:', error)
	})

const functions = require('firebase-functions');

exports.main = functions.https.onRequest((request, response) => {
 response.send({"message": '"Hello from Firebase! Lalit Anurag"'});
});

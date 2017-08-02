import {SERVER_BASE_URL as baseUrl}  from './config';

export function urls() {
	return {
		baseUrl: baseUrl,
		get_terms: `${baseUrl}/terms`,
		get_search_text: `${baseUrl}/result_text`,
		get_search_image: `${baseUrl}/result_image`
	};
};

export async function get(url, params) {
	let request = new Request(url + '?' + params, {
		method: 'GET'
	});
	
	let response = await fetch(request);

	try {
		let json = await response.json();
		return json;
	} catch(err) {
		return new Error(err);
	}
}

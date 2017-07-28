import React, { Component } from 'react';
import './App.css';
import {GridList, GridTile, AutoComplete, Table, TableBody, TableHeader, TableHeaderColumn, TableRow, TableRowColumn, Tabs, Tab} from 'material-ui';
import {get, urls} from './rest';

const styles = {
	titleWidth: {
		width: '130px'
	}, gridList: {
		overflowY: 'auto',
		margin: 'auto',
		display: 'flex',
		alignItems: 'center'
	}
};

class App extends Component {
	
	constructor(props) {
		super(props);
		this.state = {
			selected_tab: 'images',
			terms: [],
			text_results: [],
			image_results: []
		};
	}

	async onUpdateValue(term) {
		let json = await get(urls().get_terms, 'q=' + term);
		this.setState({terms: json.filter((x, i) => i % 2 === 0)});
	}
	
	async onTermSelected(item) {
		if (item) {
			let id = item.id;

			// tricky part
			// there are two records for each term search, the lower event number is the one used in result_text and the odd number (even number + 1) is the later number.
			let id_text = id % 2 === 1 ? id : id - 1;
			//\
			
			let textResults = await get(urls().get_search_text, 'id_term_search=' + id_text);
			this.setState({text_results: textResults});

			let imageResults = await get(urls().get_search_image, 'id_term_search=' + (id_text + 1));
			this.setState({image_results: imageResults}, _ => console.log(id_text + 1, this.state.image_results));
		}
	}

	onTabChanged(value) {
		this.setState({selected_tab: value});
	}
	
	render() {
		return (
			<div className="App">
			  <div style={{padding: '3px'}}>
				<AutoComplete dataSource={this.state.terms} dataSourceConfig={{text: 'term', value: 'id'}} onUpdateInput={this.onUpdateValue.bind(this)} hintText="Search Term" onNewRequest={this.onTermSelected.bind(this)} fullWidth={true}/>
			  </div>
			  
			  <Tabs value={this.state.selected_tab} onChange={this.onTabChanged.bind(this)}>
				<Tab label="Text" value="text">
				  <Table>
					<TableHeader adjustForCheckbox={false} displaySelectAll={false}>
					  <TableRow>
						<TableHeaderColumn style={styles.titleWidth}>Title</TableHeaderColumn>
						<TableHeaderColumn>Description</TableHeaderColumn>
					  </TableRow>
					</TableHeader>
					<TableBody displayRowCheckbox={false}>
					  {
						  this.state.text_results.map((x, i) => {
							  return  (
								  <TableRow key={i}>
									<TableRowColumn style={styles.titleWidth}>{x.result_title}</TableRowColumn>
									<TableRowColumn>{x.result_description}</TableRowColumn>
								  </TableRow>);
						  })
					  }


			</TableBody>
				</Table>
				</Tab>
				<Tab label="Images" value="images">

				<GridList style={styles.gridList} cols={4}>
				{
					this.state.image_results.map((x, i) => {
						return (
							<GridTile style={{maxWidth: '300px'}}>
							  <img src={urls().baseUrl + '/assets/images/' + x.filename}/>
							</GridTile>
						);
					})
				}
			</GridList>
			</Tab>
				</Tabs>
				</div>
		);
	}
}

export default App;

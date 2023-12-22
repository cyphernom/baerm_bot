import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from io import BytesIO
import os
from scipy.stats import binom,  norm

import telebot
matplotlib.use('Agg')
BOT_TOKEN = os.environ.get('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)

cache = {'data': None, 'last_updated': None}
CACHE_DURATION = 12 * 3600  # 12 hours in seconds



class DataPreprocessing:
    def __init__(self, url):
        self.url = url
        self.df = None
        self.coefs = [2.92385283e-02, 9.96869586e-01, -4.26562358e-04]
        self.n=0
        
    # Function to calculate YHATs using hardcoded coefficients
    def calculate_YHAT(self):
        self.df['YHAT'] = self.df['logprice']
        for i in range(self.df.index[0] + 1, self.df.index[-1] + 1):
            self.df.at[i, 'YHAT'] = self.coefs[0] + self.coefs[1] * self.df.at[i - 1, 'YHAT'] + self.coefs[2] * self.df.at[i, 'phaseplus']
        
    def calculate_YHATs2(self):
        self.df['decayfunc'] = 3 * np.exp(-0.0004 * self.n) * np.cos(0.005 * self.n - 1)
        self.df['YHATs2'] = self.df['YHAT'] + self.df['decayfunc']
        self.df['eYHAT'] = np.exp(self.df['YHAT'])
        self.df['eYHATs2'] = np.exp(self.df['YHATs2'])
        
    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.url, low_memory=False)
        self.df['date'] = pd.to_datetime(self.df['time'], format='%Y-%m-%d')
        self.df.drop('time', axis=1, inplace=True)
        self.df = pd.concat([self.df, pd.DataFrame({'date': pd.date_range(self.df['date'].max() + pd.Timedelta(days=1), periods=10000, freq='D')})], ignore_index=True)
        self.df.loc[self.df['BlkCnt'].isnull(), 'BlkCnt'] = 6 * 24
        self.df['sum_blocks'] = self.df['BlkCnt'].cumsum()
        self.df['hp_blocks'] = self.df['sum_blocks'] % 210001
        self.df['hindicator'] = (self.df['hp_blocks'] < 200) & (self.df['hp_blocks'].shift(1) > 209000)
        self.df['epoch'] = self.df['hindicator'].cumsum()
        self.df['reward'] = 50 / (2 ** self.df['epoch'].astype(float))
        self.df.loc[self.df['epoch'] >= 33, 'reward'] = 0
        self.df['daily_reward'] = self.df['BlkCnt'] * self.df['reward']
        self.df['tsupply'] = self.df['daily_reward'].cumsum()
        self.df['logprice'] = np.log(self.df['PriceUSD'])
        self.df['days_since_start'] = (self.df['date'] - pd.to_datetime('2008-10-31')).dt.days
        self.df['phaseplus'] = self.df['reward'] - (self.df['epoch'] + 1) ** 2
        self.df = self.df[self.df['date'] >= dt.datetime.strptime('2010-07-18', '%Y-%m-%d')].reset_index(drop=True)
        self.n = self.df.index.to_numpy()
        self.calculate_YHAT()
        self.calculate_YHATs2()


    
class Visualisation:
    def __init__(self, df):
        self.df = df
        self.is_log_scale = False

    @staticmethod
    def format_dollars(value, pos, is_minor=False):
        if value >= 1e9:
            s = '${:,.1f}B'.format(value * 1e-9)
        elif value >= 1e6:
            s = '${:,.1f}M'.format(value * 1e-6)
        elif value >= 1e3:
            s = '${:,.0f}K'.format(value * 1e-3)
        else:
            s = '${:,.0f}'.format(value)
        return s

    def plot_charts(self,time_range=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate residuals and likelihood thresholds
        residuals = self.df['logprice'] - np.log(self.df['eYHATs2'])
        residuals_std = np.std(residuals)
        likelihoods = np.arange(0.3, .7, 0.01)
        thresholds = norm.ppf(likelihoods, loc=0, scale=residuals_std)

        # Calculate upper and lower bounds for each likelihood threshold
        ub_lines = np.log(self.df['eYHATs2']).to_numpy()[:, np.newaxis] + thresholds
        lb_lines = np.log(self.df['eYHATs2']).to_numpy()[:, np.newaxis] - thresholds

        # Plot actual prices and eYHATs2
        ax.plot(self.df['date'], self.df['PriceUSD'], label='PriceUSD')
        ax.plot(self.df['date'], self.df['eYHATs2'], label='Damped BAERM', color='red')
        ax.plot(self.df['date'], self.df['eYHAT'], label='Base BAERM', color='green')
        
        # Plot the bands
        colormap = cm.get_cmap('rainbow')

        num_colors = len(likelihoods)
        color_values = np.linspace(0, 1, num_colors)

        for i in range(num_colors - 1):
            ax.fill_between(
                self.df['date'],
                np.exp(ub_lines[:, i]),
                np.exp(ub_lines[:, i + 1]),
                color=colormap(color_values[i]),
                alpha=0.3
            )
            ax.fill_between(
                self.df['date'],
                np.exp(lb_lines[:, i]),
                np.exp(lb_lines[:, i + 1]),
                color=colormap(color_values[i]),
                alpha=0.3
            )

        # Set axis scale and labels
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_dollars))
        ax.set_xlabel('Date')
        ax.set_ylabel('Exchange Rate')
        ax.legend()

        
        # Add dashed vertical lines at the change of each epoch
        epoch_changes = self.df[self.df['hindicator'] == True]
        for date in epoch_changes['date']:
            ax.axvline(x=date, color='blue', linestyle='-', linewidth=2)

        minor_locator = ticker.LogLocator(subs=(0.2, 0.4, 0.6, 0.8))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.tick_params(axis='y', which='minor', labelsize=10)

        # Add grid lines to the plot
        
        ax.grid(True)

        # Set the start date based on the input format (e.g., "12m" for 12 months)

        # Default time range: last three months to today
        start_date = pd.to_datetime('today') - pd.DateOffset(months=3)
        end_date = pd.to_datetime('today')

        if time_range is not None:
            parts = time_range.split()
            if len(parts) == 1:
                start_date, end_date = self.parse_time_range(parts[0], end_date)
            elif len(parts) == 2:
                start_date, _ = self.parse_time_range(parts[0], end_date)
                _, end_date = self.parse_time_range(parts[1], end_date)

        ax.set_xlim([start_date, end_date])
        # Calculate and set the y-axis limit
        y_max = self.calculate_max_y_within_range(start_date, end_date, ub_lines, lb_lines)
        ax.set_ylim([0, y_max])

        buf = BytesIO()
        plt.savefig(buf, format='png')
        # Close the figure to free up memory
        plt.close(fig)
        buf.seek(0)
        return buf  # Returns a buffer with the plot image

    def parse_time_range(self, part, default_date):
        if part.startswith('-') and part.endswith('m'):
            months = int(part[1:-1])
            return default_date - pd.DateOffset(months=months), default_date
        elif part.endswith('m'):
            months = int(part[:-1])
            return default_date, default_date + pd.DateOffset(months=months)
        else:
            return pd.to_datetime(part), default_date


    def calculate_max_y_within_range(self, start_date, end_date, ub_lines, lb_lines):
        # Filter dates within the specified range
        mask = (self.df['date'] >= start_date) & (self.df['date'] <= end_date)
        filtered_dates = self.df['date'][mask]

        # Find the indices of these dates in the DataFrame
        indices = filtered_dates.index

        # Calculate the maximum value among the upper and lower bounds within this range
        max_value = max(np.exp(ub_lines[indices].max()), np.exp(lb_lines[indices].max()))

        # Set the y-axis limit to be 10% higher than this maximum
        y_max = max_value * 1.1

        return y_max

    def plot_z_score_chart(self, time_range=None, model_type='damped'):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set default time range to last three months if not specified
        if time_range is None:
            start_date = pd.to_datetime('today') - pd.DateOffset(months=3)
            end_date = pd.to_datetime('today')
            date_range_title = "Last 3 Months"
        else:
            start_date, end_date = self.parse_time_range(time_range, pd.to_datetime('today'))
            date_range_title = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
         
                 # Adjust start date if it is before the first date of epoch >= 2
        first_valid_date = self.df[self.df['epoch'] >= 2]['date'].min()
        if start_date < first_valid_date:
            start_date = first_valid_date
            date_range_title = f"Adjusted: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        model_column = 'eYHATs2' if model_type == 'damped' else 'eYHAT'      
        filtered_df = self.df[self.df['epoch'] >= 2]
        residuals = filtered_df['logprice'] - np.log(filtered_df[model_column])
        residuals_std = residuals.std()
            
        # Filter the DataFrame based on the time range
        mask = (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
        filtered_df = filtered_df[mask]

        
        # Check if the filtered DataFrame is not empty
        if not filtered_df.empty:


            

            filtered_df['z_scores'] = residuals / residuals_std

            # Plot Z-score zones as shaded areas
            ax.fill_between(filtered_df['date'], -1, 1, color='palegreen', alpha=0.3)
            ax.fill_between(filtered_df['date'], -2, -1, color='palegoldenrod', alpha=0.3)
            ax.fill_between(filtered_df['date'], 1, 2, color='palegoldenrod', alpha=0.3)
            ax.fill_between(filtered_df['date'], -3, -2, color='salmon', alpha=0.3)
            ax.fill_between(filtered_df['date'], 2, 3, color='salmon', alpha=0.3)

            # Plot Z-scores
            ax.plot(filtered_df['date'], filtered_df['z_scores'], label='Z-Scores', color='blue', linewidth=2)

            # Improved chart aesthetics
            ax.grid(True, which='both', linestyle='-', linewidth=0.5)
            ax.set_facecolor('whitesmoke')

            # Set axis labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Z-Score')
            ax.set_title(f'Z-Score Variation Chart ({date_range_title})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available for the selected range', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes)
                    
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf


    def plot_macd_with_predictions(self, time_range=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Default values
        start_date_actual = pd.to_datetime('today') - pd.DateOffset(months=3)
        end_date_actual = pd.to_datetime('today')
        end_date_predict = end_date_actual + pd.DateOffset(months=3)

        if time_range is not None:
            parts = time_range.split()
            print(f"{parts}")
            if len(parts) >= 1:
                start_date_actual, _ = self.parse_time_range(parts[0], end_date_actual)
            if len(parts) >= 2:
                _, end_date_actual = self.parse_time_range(parts[0], end_date_actual)
                end_date_predict = self.parse_time_range(parts[1], end_date_actual)[1]

        
        print(f"end_date_actual:{end_date_actual}")
        print(f"end end_date_predict:{end_date_predict}")
        print(f"end start_date_actual:{start_date_actual}")   
        
        # Fetch actual data
        actual_data = self.df[(self.df['date'] >= start_date_actual) & (self.df['date'] <= end_date_actual)]

        # Determine end date for predictions
        end_date_predict = end_date_actual + pd.DateOffset(months=3) if time_range is None else self.parse_time_range(parts[-1], end_date_actual)[1]

        # Fetch predicted data
        predicted_data = self.df[(self.df['date'] > end_date_actual) & (self.df['date'] <= end_date_predict)]
        predicted_data['logprice'] = np.log(predicted_data['eYHATs2'])

        # Merge actual and predicted data
        extended_data = pd.concat([actual_data, predicted_data])

        # Calculate EMAs, MACD, and Signal line
        extended_data['ema12'] = extended_data['logprice'].ewm(span=12, adjust=False).mean()
        extended_data['ema26'] = extended_data['logprice'].ewm(span=26, adjust=False).mean()
        extended_data['macd'] = extended_data['ema12'] - extended_data['ema26']
        extended_data['signal_line'] = extended_data['macd'].ewm(span=9, adjust=False).mean()

        # Plot MACD and Signal line
        ax1.plot(extended_data['date'], extended_data['macd'], label='MACD', color='blue')
        ax1.plot(extended_data['date'], extended_data['signal_line'], label='Signal Line', color='red')
        ax1.axvspan(end_date_actual, end_date_predict, color='yellow', alpha=0.3, label='Predicted Range')
        ax1.set_ylabel('MACD Value')
        ax1.legend()
        ax1.grid(True)

        # Plot MACD histogram
        extended_data['macd_hist'] = extended_data['macd'] - extended_data['signal_line']
        ax2.bar(extended_data['date'], extended_data['macd_hist'], label='MACD Histogram', color='grey')
        ax2.axvspan(end_date_actual, end_date_predict, color='yellow', alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Histogram Value')
        ax2.legend()
        ax2.grid(True)

        # Set title
        fig.suptitle('MACD with Model Predictions and Histogram')

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf





def load_and_cache_data(url):
    global cache
    current_time = pd.Timestamp.now(tz='UTC')

    # Check if data is cached and still valid
    if cache['data'] is not None and (current_time - cache['last_updated']).total_seconds() < CACHE_DURATION:
        return cache['data']
    
    # Load new data
    data_preprocessing = DataPreprocessing(url)
    data_preprocessing.load_and_preprocess_data()
    cache['data'] = data_preprocessing.df
    cache['last_updated'] = current_time
    return cache['data']







def get_bot_data(chart, time_range=None):
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    df = load_and_cache_data(url)  # Use the new function

    if chart:
        visualisation = Visualisation(df)
        chart_buffer = visualisation.plot_charts(time_range)
        return chart_buffer
    else:
        return df['eYHATs2'].iloc[-7:]


@bot.message_handler(commands=['macdpred'])
def get_macd_chart(message):
    try:

        command_parts = message.text.split()
        time_range = None

        # Check the number of arguments and set the time_range accordingly
        if len(command_parts) == 2:
            # Single argument (e.g., "12m" or "-12m")
            time_range = command_parts[1]
        elif len(command_parts) == 3:
            # Two arguments (e.g., "-12m 12m")
            time_range = f"{command_parts[1]} {command_parts[2]}"
                
        url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
        df = load_and_cache_data(url) 
        visualisation = Visualisation(df)
        chart_buffer = visualisation.plot_macd_with_predictions(time_range)
        bot.send_photo(message.chat.id, chart_buffer)
        bot.send_message(message.chat.id, "Lightening Tip: btconometrics@getalby.com")

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")

@bot.message_handler(commands=['zscorechart'])
def get_z_score_chart(message):
    try:
        command_parts = message.text.split()
        time_range = None
        model_type = 'damped'  # Default model type

        # Parse time range and model type from the command
        if len(command_parts) > 1:
            time_range = command_parts[1]
            if len(command_parts) > 2:
                model_type = command_parts[2]
                
        url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
        df = load_and_cache_data(url) 
        visualisation = Visualisation(df)
        chart_buffer = visualisation.plot_z_score_chart(time_range,model_type)
        bot.send_photo(message.chat.id, chart_buffer)
        bot.send_message(message.chat.id, "Lightening Tip: btconometrics@getalby.com")

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")


@bot.message_handler(commands=['chart'])
def get_chart(message):
    try:
        command_parts = message.text.split()
        time_range = None

        # Check the number of arguments and set the time_range accordingly
        if len(command_parts) == 2:
            # Single argument (e.g., "12m" or "-12m")
            time_range = command_parts[1]
        elif len(command_parts) == 3:
            # Two arguments (e.g., "-12m 12m")
            time_range = f"{command_parts[1]} {command_parts[2]}"

        chart_buffer = get_bot_data(chart=True, time_range=time_range)
        chart_buffer.seek(0)
        bot.send_photo(message.chat.id, chart_buffer)
        bot.send_message(message.chat.id, "Lightening Tip: btconometrics@getalby.com")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")






@bot.message_handler(commands=['help', 'start', 'hello'])
def send_help(message):
    help_message = """
    Welcome to the Baerm Bot! Here are the available commands and their usage:

    - /start or /hello or /help: Start a conversation with the bot, display this message
    
    - /chart [time_range]: Generate a BAERM chart for the last 3 months. You can specify the time range like "/chart 12m" (default is 3 months). Example usage: "/chart 6m" or "/chart -12m 12m".
    
    - /zscorechart [starting_from] [base, damped]: Generate a chart of the Z score for the price from the starting_from date to today. Defaults to the damped model, but you can specify base i.e. /zscorechart -12m base

    - /likelihood [custom_price] [base, (default) damped]: Calculate likelihood and compare it to the actual price. You can provide a custom price for comparison (optional). Example usage: "/likelihood" or "/likelihood 50000.0".
    
    - /baermline [num_days]: Get the BAERM line data for the next 7 days (default). You can specify the number of days to retrieve. Example usage: "/baermline" or "/baermline 14".
    
    Data is pulled in from coinmetrics and can be a day or so old. The bot's cache is updated every 12hours.

    For additional assistance, contact me via email at: btconometrics@protonmail.com. 
    
    If you like this bot and want me to work on it more encourage me by sending me some sats at: btconometrics@getalby.com
    """
    bot.reply_to(message, help_message)


@bot.message_handler(commands=['likelihood'])
def get_likelihood(message):
    try:
        command_parts = message.text.split()
        custom_price = None
        model_type = 'damped'  # Default model type

        # Loop through the command parts to parse arguments
        for part in command_parts[1:]:
            if part.isdigit():
                custom_price = float(part)
            elif part in ['base', 'damped']:
                model_type = part

        data_date, actual_price, forecast_price, z_score, interpretation, date_explainer = calculate_likelihood_and_compare(custom_price, model_type)

        if actual_price is not None:
            response_message = (f"Likelihood calculated for: {data_date}{date_explainer}\n"
                                f"Price used for calculation: ${actual_price:.2f}\n"
                                f"Forecasted price: ${forecast_price:.2f}\n"
                                f"Z-score: {z_score:.2f} - {interpretation}")
        else:
            response_message = "No recent data available."

        bot.reply_to(message, response_message)
        bot.send_message(message.chat.id, "Lightening Tip: btconometrics@getalby.com")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")


def calculate_likelihood_and_compare(custom_price=None, model_type='damped'):
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    df = load_and_cache_data(url)
    model_column = 'eYHATs2' if model_type == 'damped' else 'eYHAT'
    for days_behind in range(1, 4):
        check_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=days_behind)
        check_data = df[df['date'] == check_date]

        if not check_data.empty:
            logprice = check_data['logprice'].values[0]
            if not np.isnan(logprice):
                today_forecast = np.exp(np.log(check_data[model_column].values[0]))
                price_to_use = custom_price if custom_price is not None else np.exp(logprice)
                data_date = check_date.strftime("%Y-%m-%d")
                date_explainer = "" if days_behind == 1 else f" (using data from {data_date} as more recent data is unavailable)"
                
                # Filter the DataFrame to include only rows where 'epoch' is greater than 2
                filtered_df = df[df['epoch'] >=2]
                
                # Calculate the residuals and Z-score for the filtered data
                residuals = filtered_df['logprice'] - np.log(filtered_df['eYHATs2'])
   
                residuals_std = np.std(residuals)
                z_score = (np.log(price_to_use) - np.log(today_forecast)) / residuals_std
                interpretation = interpret_z_score(z_score)
                
                return data_date, price_to_use, today_forecast, z_score, interpretation, date_explainer

    return None, None, None, None, "No recent data available."




def interpret_z_score(z):
    if abs(z) < 1:
        return "typical (within 1 std dev)"
    elif abs(z) < 2:
        return "unusual (between 1 and 2 std devs)"
    else:
        return "very unusual (more than 2 std devs)"





@bot.message_handler(commands=['baermline'])
def get_baermline(message):
    try:
        command_parts = message.text.split()
        # Default to 7 days if no argument is provided
        num_days = 7 
        if len(command_parts) > 1:
            num_days = int(command_parts[1])

        # Fetch the BAERM line data
        forecast_data = get_baermline_data(num_days)
        formatted_message = forecast_data.to_string(index=False)
        bot.reply_to(message, formatted_message + "\nLightening Tip: btconometrics@getalby.com")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")

def get_baermline_data(num_days):
    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    df = load_and_cache_data(url)  # Use the new function

    # Find today's date in the DataFrame
    today = pd.to_datetime('today').normalize()
    future_date = today + pd.Timedelta(days=num_days)

    # Get the range of data from today to the future date
    forecast_range = df[(df['date'] >= today) & (df['date'] < future_date)]
    return forecast_range['eYHATs2'].round(2)

@bot.message_handler(commands=['arbitrage'])
def get_arbitrage_opportunities(message):
    try:
        df = load_and_cache_data("https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv")
        
        # Filter DataFrame for out-of-sample data
        mask = df['epoch'] >= 2
        filtered_df = df[mask].copy()  # Make a copy to avoid SettingWithCopyWarning

        # Calculate residuals and standard deviation
        filtered_df.loc[:, 'residuals'] = filtered_df['logprice'] - np.log(filtered_df['eYHATs2'])
        residuals_std = filtered_df['residuals'].std()

        # Calculate likelihood using the PDF of the normal distribution
        filtered_df.loc[:, 'likelihood'] = filtered_df.apply(
            lambda row: norm.pdf(row['logprice'], loc=np.log(row['eYHATs2']), scale=residuals_std), axis=1)

        # Define a threshold for likelihood
        likelihood_threshold = 0.01  # Example threshold

        # Find arbitrage opportunities
        opportunities = filtered_df[filtered_df['likelihood'] < likelihood_threshold]

        if not opportunities.empty:
            response = "Potential Arbitrage Opportunities:\n"
            for _, row in opportunities.iterrows():
                response += f"Date: {row['date']}, Actual Price: {row['PriceUSD']}, Predicted Price: {row['eYHATs2']}, Likelihood: {row['likelihood']:.5f}\n"
        else:
            response = "No significant arbitrage opportunities based on likelihood."

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")




if __name__ == "__main__":
    # Initialize cache
    load_and_cache_data("https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv")
    
    bot.infinity_polling()

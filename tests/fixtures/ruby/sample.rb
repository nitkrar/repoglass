class RefundProcessor
  def process(order, retries: 3)
    validate(order)
    Ledger.record(order, retries)
  end

  def validate(order)
    raise ArgumentError unless order.total.positive?
  end
end
